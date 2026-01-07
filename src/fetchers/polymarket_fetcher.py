"""Fetch prices from Polymarket for EPL markets."""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PolymarketPrices:
    """Polymarket prices for a match."""
    match_id: str
    home_team: str
    away_team: str

    # 1X2 prices (0-1, where 0.55 = 55 cents = 55% implied)
    home_win_price: float
    draw_price: float
    away_win_price: float

    # O/U prices
    over_1_5_price: Optional[float] = None
    under_1_5_price: Optional[float] = None
    over_2_5_price: Optional[float] = None
    under_2_5_price: Optional[float] = None
    over_3_5_price: Optional[float] = None
    under_3_5_price: Optional[float] = None

    # BTTS prices
    btts_yes_price: Optional[float] = None
    btts_no_price: Optional[float] = None

    # Liquidity (available at best price)
    home_win_liquidity: float = 0
    over_2_5_liquidity: float = 0
    btts_yes_liquidity: float = 0

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PolymarketFetcher:
    """Fetch EPL market prices from Polymarket.

    Uses the Polymarket Gamma API for market discovery and CLOB for prices.
    """

    # Polymarket API endpoints
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    # EPL team names for search
    EPL_TEAMS = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Tottenham",
        "West Ham", "Wolves"
    ]

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def fetch_epl_markets(self) -> list[PolymarketPrices]:
        """Fetch all EPL match markets."""
        try:
            # Search for Premier League markets
            markets = await self._search_markets("Premier League")

            results = []
            for market in markets:
                prices = await self._extract_prices(market)
                if prices:
                    results.append(prices)

            logger.info(f"Fetched {len(results)} EPL markets from Polymarket")
            return results

        except Exception as e:
            logger.error(f"Error fetching Polymarket data: {e}")
            return []

    async def _search_markets(self, query: str) -> list[dict]:
        """Search for markets by query using Gamma API."""
        try:
            response = await self.client.get(
                f"{self.GAMMA_API}/markets",
                params={
                    "closed": "false",
                    "tag_slug": "sports",
                    "limit": 100,
                }
            )
            response.raise_for_status()
            all_markets = response.json()

            # Filter for EPL markets
            epl_markets = []
            for market in all_markets:
                question = market.get("question", "").lower()
                description = market.get("description", "").lower()

                # Check if it's a Premier League market
                if "premier league" in question or "premier league" in description:
                    epl_markets.append(market)
                    continue

                # Check if it mentions EPL teams
                for team in self.EPL_TEAMS:
                    if team.lower() in question:
                        epl_markets.append(market)
                        break

            return epl_markets

        except Exception as e:
            logger.error(f"Market search error: {e}")
            return []

    async def _extract_prices(self, market: dict) -> Optional[PolymarketPrices]:
        """Extract prices from market data."""
        try:
            question = market.get("question", "")
            condition_id = market.get("condition_id", "")

            if not condition_id:
                return None

            # Parse team names from market question
            home_team, away_team = self._parse_teams(question)

            if not home_team or not away_team:
                return None

            # Get tokens/outcomes
            tokens = market.get("tokens", [])
            if not tokens:
                return None

            # Extract prices from tokens
            home_price = draw_price = away_price = 0.0
            home_liquidity = 0.0

            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = float(token.get("price", 0))

                if "home" in outcome or home_team.lower() in outcome:
                    home_price = price
                    home_liquidity = float(token.get("liquidity", 0))
                elif "draw" in outcome:
                    draw_price = price
                elif "away" in outcome or away_team.lower() in outcome:
                    away_price = price

            # Skip if no valid prices
            if home_price == 0 and draw_price == 0 and away_price == 0:
                return None

            return PolymarketPrices(
                match_id=condition_id,
                home_team=home_team,
                away_team=away_team,
                home_win_price=home_price,
                draw_price=draw_price,
                away_win_price=away_price,
                home_win_liquidity=home_liquidity,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.warning(f"Error extracting prices: {e}")
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

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class PolymarketDirectFetcher:
    """Alternative direct HTTP fetcher for Polymarket sports.

    Fallback if CLOB/Gamma APIs don't provide needed data.
    """

    SPORTS_API = "https://polymarket.com/api/events"

    async def fetch_epl_games(self) -> list[dict]:
        """Fetch EPL games from sports API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    self.SPORTS_API,
                    params={
                        "tag": "sports",
                        "active": "true",
                    },
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()

                events = response.json()
                epl_events = []

                for event in events:
                    title = event.get("title", "").lower()
                    if "premier league" in title or "epl" in title:
                        epl_events.append(event)

                return epl_events

            except Exception as e:
                logger.error(f"Error fetching from sports API: {e}")
                return []
