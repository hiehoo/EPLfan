"""Fetch Polymarket prices via DomeAPI."""
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DomeApiPrices:
    """DomeAPI Polymarket prices for a match."""
    market_slug: str
    condition_id: str
    home_team: str
    away_team: str

    # 1X2 prices (0-1, where 0.55 = 55% implied)
    home_win_price: float
    draw_price: float
    away_win_price: float

    # O/U prices (optional)
    over_1_5_price: Optional[float] = None
    under_1_5_price: Optional[float] = None
    over_2_5_price: Optional[float] = None
    under_2_5_price: Optional[float] = None
    over_3_5_price: Optional[float] = None
    under_3_5_price: Optional[float] = None

    # BTTS prices (optional)
    btts_yes_price: Optional[float] = None
    btts_no_price: Optional[float] = None

    # Market metrics
    volume: float = 0
    liquidity: float = 0

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DomeApiFetcher:
    """Fetch EPL market prices from Polymarket via DomeAPI.

    DomeAPI provides a documented wrapper around Polymarket with
    reliable rate limits and structured responses.
    """

    BASE_URL = "https://api.domeapi.io/v1"

    # EPL team names for search
    EPL_TEAMS = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Tottenham",
        "West Ham", "Wolves"
    ]

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

    async def fetch_epl_markets(self) -> list[DomeApiPrices]:
        """Fetch all EPL match markets from Polymarket via DomeAPI.

        Returns:
            List of DomeApiPrices with 1X2/OU/BTTS prices
        """
        if not self._api_key:
            logger.warning("DomeAPI not configured, returning empty list")
            return []

        try:
            # Search for Premier League markets
            markets = await self._search_markets("premier league")

            results = []
            for market in markets:
                prices = self._extract_prices(market)
                if prices:
                    results.append(prices)

            logger.info(f"Fetched {len(results)} EPL markets from DomeAPI")
            return results

        except Exception as e:
            logger.error(f"Error fetching DomeAPI data: {e}")
            return []

    async def _search_markets(self, query: str) -> list[dict]:
        """Search for markets by query.

        Args:
            query: Search term (e.g., "premier league")

        Returns:
            List of market dictionaries
        """
        await self._respect_rate_limit()

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/polymarket/markets",
                params={"q": query, "closed": "false"}
            )
            response.raise_for_status()

            data = response.json()
            markets = data.get("markets", data) if isinstance(data, dict) else data

            # Filter for EPL-relevant markets
            epl_markets = []
            for market in markets if isinstance(markets, list) else []:
                question = market.get("question", "").lower()
                description = market.get("description", "").lower()

                # Check for Premier League keywords
                if "premier league" in question or "premier league" in description:
                    epl_markets.append(market)
                    continue

                # Check for EPL team names
                for team in self.EPL_TEAMS:
                    if team.lower() in question:
                        epl_markets.append(market)
                        break

            return epl_markets

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("DomeAPI rate limit hit, backing off")
                await asyncio.sleep(10)
            logger.error(f"DomeAPI search error: {e}")
            return []
        except Exception as e:
            logger.error(f"DomeAPI search error: {e}")
            return []

    def _extract_prices(self, market: dict) -> Optional[DomeApiPrices]:
        """Extract prices from market data.

        Args:
            market: Market dictionary from DomeAPI

        Returns:
            DomeApiPrices if valid match market, None otherwise
        """
        try:
            question = market.get("question", "")
            market_slug = market.get("market_slug", market.get("slug", ""))
            condition_id = market.get("condition_id", "")

            # Parse team names
            home_team, away_team = self._parse_teams(question)
            if not home_team or not away_team:
                return None

            # Extract token/outcome prices
            tokens = market.get("tokens", market.get("outcomes", []))
            if not tokens:
                return None

            home_price = draw_price = away_price = 0.0
            over_2_5 = under_2_5 = btts_yes = btts_no = None
            volume = market.get("volume", 0)
            liquidity = market.get("liquidity", 0)

            for token in tokens:
                outcome = token.get("outcome", token.get("name", "")).lower()
                price = float(token.get("price", 0))

                # Match result (1X2)
                if "home" in outcome or home_team.lower() in outcome:
                    home_price = price
                elif "draw" in outcome:
                    draw_price = price
                elif "away" in outcome or away_team.lower() in outcome:
                    away_price = price

                # Over/Under 2.5
                elif "over 2.5" in outcome or "over2.5" in outcome:
                    over_2_5 = price
                elif "under 2.5" in outcome or "under2.5" in outcome:
                    under_2_5 = price

                # BTTS
                elif "btts yes" in outcome or "both teams to score" in outcome:
                    btts_yes = price
                elif "btts no" in outcome:
                    btts_no = price

            # Skip if no valid prices
            if home_price == 0 and draw_price == 0 and away_price == 0:
                return None

            return DomeApiPrices(
                market_slug=market_slug,
                condition_id=condition_id,
                home_team=home_team,
                away_team=away_team,
                home_win_price=home_price,
                draw_price=draw_price,
                away_win_price=away_price,
                over_2_5_price=over_2_5,
                under_2_5_price=under_2_5,
                btts_yes_price=btts_yes,
                btts_no_price=btts_no,
                volume=volume,
                liquidity=liquidity,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.warning(f"Error extracting DomeAPI prices: {e}")
            return None

    def _parse_teams(self, question: str) -> tuple[str, str]:
        """Parse team names from market question.

        Handles patterns like:
        - "Arsenal vs Chelsea"
        - "Will Arsenal win against Chelsea?"
        - "Arsenal v Chelsea - Winner"
        """
        # Pattern: Team1 vs/v./versus Team2
        vs_pattern = r"(.+?)\s+(?:vs\.?|v\.?|versus)\s+(.+?)(?:\s*[-:?]|$)"
        match = re.search(vs_pattern, question, re.IGNORECASE)

        if match:
            home = match.group(1).strip()
            away = match.group(2).strip()
            # Clean up common suffixes
            away = re.sub(r"\s*[-:?].*$", "", away)
            return home, away

        # Pattern: Will Team1 win/beat Team2
        win_pattern = r"Will\s+(.+?)\s+(?:win|beat)\s+(?:against\s+)?(.+?)(?:\?|$)"
        match = re.search(win_pattern, question, re.IGNORECASE)

        if match:
            return match.group(1).strip(), match.group(2).strip()

        return "", ""

    async def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits (Free tier: 1/sec)."""
        async with self._rate_limiter:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self._last_request = time.time()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
