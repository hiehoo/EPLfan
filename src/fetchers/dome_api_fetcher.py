"""Fetch Polymarket prices via DomeAPI."""
import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


# Team name to Polymarket slug code mapping
TEAM_SLUG_MAP = {
    "Arsenal": "ars",
    "Aston Villa": "avl",
    "Bournemouth": "bou",
    "Brentford": "bre",
    "Brighton": "bha",
    "Brighton & Hove Albion": "bha",
    "Chelsea": "che",
    "Crystal Palace": "cry",
    "Everton": "eve",
    "Fulham": "ful",
    "Ipswich": "ips",
    "Ipswich Town": "ips",
    "Leicester": "lei",
    "Leicester City": "lei",
    "Liverpool": "liv",
    "Man City": "mci",
    "Manchester City": "mci",
    "Man United": "mun",
    "Manchester United": "mun",
    "Newcastle": "new",
    "Newcastle United": "new",
    "Nott'm Forest": "nfo",
    "Nottingham Forest": "nfo",
    "Southampton": "sou",
    "Tottenham": "tot",
    "Tottenham Hotspur": "tot",
    "West Ham": "whu",
    "West Ham United": "whu",
    "Wolves": "wol",
    "Wolverhampton": "wol",
    "Wolverhampton Wanderers": "wol",
}

# Full team names on Polymarket
TEAM_FULL_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Ipswich": "Ipswich Town",
    "Ipswich Town": "Ipswich Town",
    "Leicester": "Leicester City",
    "Leicester City": "Leicester City",
    "Liverpool": "Liverpool FC",
    "Man City": "Manchester City",
    "Manchester City": "Manchester City",
    "Man United": "Manchester United",
    "Manchester United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Southampton": "Southampton FC",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "West Ham United": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
}


@dataclass
class PolymarketMatchPrices:
    """Polymarket prices for a specific match."""
    home_team: str
    away_team: str
    match_date: str  # YYYY-MM-DD

    # Win prices (0-1 probability)
    home_win_price: float = 0.0
    away_win_price: float = 0.0
    # Draw is typically 1 - home - away (Polymarket uses Yes/No for each team)

    # Market metadata
    home_market_slug: str = ""
    away_market_slug: str = ""
    volume: float = 0.0

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DomeApiFetcher:
    """Fetch EPL match prices from Polymarket via DomeAPI.

    Flow:
    1. Get fixture (home_team, away_team, date) from Football-Data/Betfair
    2. Search DomeAPI for "Will {Team} win on {date}?" markets
    3. Fetch prices via /polymarket/market-price/{token_id}
    """

    BASE_URL = "https://api.domeapi.io/v1"

    def __init__(self):
        if not settings.dome_api.is_configured():
            logger.warning("DomeAPI key not configured - fetcher will not work")
            self._api_key = ""
        else:
            self._api_key = settings.dome_api.api_key.get_secret_value()

        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        )
        # Rate limiter for Free tier (1 req/sec)
        self._rate_limiter = asyncio.Semaphore(1)
        self._last_request = 0.0

        # Cache for markets to avoid repeated searches
        self._market_cache: dict[str, dict] = {}

    async def fetch_match_prices(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Optional[PolymarketMatchPrices]:
        """Fetch Polymarket prices for a specific match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match kickoff datetime

        Returns:
            PolymarketMatchPrices if markets found, None otherwise
        """
        if not self._api_key:
            logger.debug("DomeAPI not configured")
            return None

        date_str = match_date.strftime("%Y-%m-%d")

        # Search for both team win markets
        home_market = await self._find_team_win_market(home_team, date_str)
        away_market = await self._find_team_win_market(away_team, date_str)

        if not home_market and not away_market:
            logger.debug(f"No Polymarket markets for {home_team} vs {away_team} on {date_str}")
            return None

        prices = PolymarketMatchPrices(
            home_team=home_team,
            away_team=away_team,
            match_date=date_str,
        )

        # Get home win price
        if home_market:
            prices.home_market_slug = home_market.get("market_slug", "")
            token_yes = home_market.get("side_a", {}).get("id")
            if token_yes:
                price = await self._get_token_price(token_yes)
                if price is not None:
                    prices.home_win_price = price
            prices.volume += home_market.get("volume_total", 0)

        # Get away win price
        if away_market:
            prices.away_market_slug = away_market.get("market_slug", "")
            token_yes = away_market.get("side_a", {}).get("id")
            if token_yes:
                price = await self._get_token_price(token_yes)
                if price is not None:
                    prices.away_win_price = price
            prices.volume += away_market.get("volume_total", 0)

        return prices

    async def _find_team_win_market(self, team: str, date_str: str) -> Optional[dict]:
        """Find "Will {Team} win on {date}?" market.

        Args:
            team: Team name
            date_str: Date in YYYY-MM-DD format

        Returns:
            Market dict if found, None otherwise
        """
        cache_key = f"{team}:{date_str}"
        if cache_key in self._market_cache:
            return self._market_cache[cache_key]

        # Get Polymarket team name
        pm_team = TEAM_FULL_NAME_MAP.get(team, team)

        # Search patterns to try
        search_queries = [
            f"{pm_team} win {date_str}",
            f"{team} win {date_str}",
        ]

        for query in search_queries:
            await self._respect_rate_limit()

            try:
                response = await self.client.get(
                    f"{self.BASE_URL}/polymarket/markets",
                    params={"q": query, "closed": "false", "limit": 10}
                )
                response.raise_for_status()

                data = response.json()
                markets = data.get("markets", [])

                for market in markets:
                    title = market.get("title", "")
                    tags = market.get("tags", [])

                    # Check if this is the right market
                    if date_str in title and ("EPL" in tags or "Premier League" in tags):
                        # Verify team name is in title
                        if pm_team.lower() in title.lower() or team.lower() in title.lower():
                            self._market_cache[cache_key] = market
                            logger.debug(f"Found market: {title}")
                            return market

            except Exception as e:
                logger.debug(f"Search error for '{query}': {e}")

        # Try tag-based search as fallback
        await self._respect_rate_limit()
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/polymarket/markets",
                params={"tags": "EPL", "closed": "false", "limit": 100}
            )
            response.raise_for_status()

            data = response.json()
            markets = data.get("markets", [])

            for market in markets:
                title = market.get("title", "")

                # Match pattern: "Will {Team} win on {date}?"
                if date_str in title:
                    if pm_team.lower() in title.lower() or team.lower() in title.lower():
                        self._market_cache[cache_key] = market
                        logger.debug(f"Found market via tags: {title}")
                        return market

        except Exception as e:
            logger.debug(f"Tag search error: {e}")

        self._market_cache[cache_key] = None
        return None

    async def _get_token_price(self, token_id: str) -> Optional[float]:
        """Get current price for a token.

        Args:
            token_id: Polymarket token ID

        Returns:
            Price (0-1) or None if not available
        """
        await self._respect_rate_limit()

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/polymarket/market-price/{token_id}"
            )

            if response.status_code == 200:
                data = response.json()
                price = data.get("price")
                if price is not None:
                    return float(price)

        except Exception as e:
            logger.debug(f"Price fetch error: {e}")

        return None

    async def fetch_epl_markets(self) -> list[dict]:
        """Fetch all available EPL markets from Polymarket.

        Returns list of market dicts with structure:
        {
            "title": "Will Liverpool FC win on 2026-01-01?",
            "market_slug": "epl-liv-lee-2026-01-01-liv",
            "game_start_time": "2026-01-01T17:30:00Z",
            "side_a": {"id": "...", "label": "Yes"},
            "side_b": {"id": "...", "label": "No"},
            ...
        }
        """
        if not self._api_key:
            logger.warning("DomeAPI not configured")
            return []

        await self._respect_rate_limit()

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/polymarket/markets",
                params={"tags": "EPL", "closed": "false", "limit": 100}
            )
            response.raise_for_status()

            data = response.json()
            markets = data.get("markets", [])

            # Filter for match-specific markets (not season winners)
            match_markets = []
            for m in markets:
                title = m.get("title", "")
                # Match pattern: contains date like "2026-01-01" or "win on"
                if re.search(r"\d{4}-\d{2}-\d{2}", title) or "win on" in title.lower():
                    match_markets.append(m)

            logger.info(f"Found {len(match_markets)} EPL match markets")
            return match_markets

        except Exception as e:
            logger.error(f"Error fetching EPL markets: {e}")
            return []

    async def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits (Free tier: 1/sec)."""
        import time
        async with self._rate_limiter:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < 1.1:  # Add small buffer
                await asyncio.sleep(1.1 - elapsed)
            self._last_request = time.time()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
