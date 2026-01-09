"""Data fetcher orchestration."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Union

from src.fetchers.betfair_fetcher import BetfairFetcher, BetfairOdds
from src.fetchers.understat_fetcher import UnderstatFetcher, TeamXGData
from src.fetchers.clubelo_fetcher import ClubELOFetcher, TeamELO
from src.fetchers.polymarket_fetcher import PolymarketFetcher, PolymarketPrices
from src.fetchers.dome_api_fetcher import DomeApiFetcher, PolymarketMatchPrices
from src.fetchers.historical_data_seeder import HistoricalDataSeeder, historical_seeder
from src.fetchers.fixture_fetcher import FixtureFetcher, Fixture, fixture_fetcher
from src.config.settings import settings
from src.storage.database import db
from src.storage.models import Match, OddsSnapshot, TeamStats

logger = logging.getLogger(__name__)

__all__ = [
    "DataOrchestrator",
    "BetfairFetcher",
    "BetfairOdds",
    "UnderstatFetcher",
    "TeamXGData",
    "ClubELOFetcher",
    "TeamELO",
    "PolymarketFetcher",
    "PolymarketPrices",
    "DomeApiFetcher",
    "PolymarketMatchPrices",
    "HistoricalDataSeeder",
    "historical_seeder",
    "FixtureFetcher",
    "Fixture",
    "fixture_fetcher",
]


class DataOrchestrator:
    """Coordinate all data fetching and storage."""

    def __init__(self):
        self.betfair = BetfairFetcher()
        self.understat = UnderstatFetcher()
        self.clubelo = ClubELOFetcher()
        self.polymarket = PolymarketFetcher()
        # Prefer DomeAPI when configured (better rate limits and structure)
        self.dome_api = DomeApiFetcher() if settings.dome_api.is_configured() else None

    async def fetch_all(self, hours_ahead: int = 48) -> dict:
        """Fetch all data sources concurrently.

        Args:
            hours_ahead: Number of hours ahead to fetch matches for

        Returns:
            Dict with keys 'betfair', 'xg', 'elo', 'polymarket' containing fetched data
        """
        logger.info("Starting full data fetch...")

        # Fetch concurrently
        betfair_task = asyncio.create_task(self._wrap_sync(
            lambda: self.betfair.fetch_upcoming_matches(hours_ahead)
        ))
        understat_task = asyncio.create_task(self.understat.fetch_all_teams())
        elo_task = asyncio.create_task(self.clubelo.fetch_epl_ratings())

        # Prefer DomeAPI for Polymarket data when configured
        if self.dome_api and settings.dome_api.is_configured():
            logger.info("Using DomeAPI for Polymarket data")
            polymarket_task = asyncio.create_task(self.dome_api.fetch_epl_markets())
        else:
            logger.info("Using native Polymarket API")
            polymarket_task = asyncio.create_task(self.polymarket.fetch_epl_markets())

        betfair_data, xg_data, elo_data, poly_data = await asyncio.gather(
            betfair_task, understat_task, elo_task, polymarket_task,
            return_exceptions=True
        )

        # Handle errors
        results = {
            "betfair": betfair_data if not isinstance(betfair_data, Exception) else [],
            "xg": xg_data if not isinstance(xg_data, Exception) else {},
            "elo": elo_data if not isinstance(elo_data, Exception) else {},
            "polymarket": poly_data if not isinstance(poly_data, Exception) else [],
        }

        for source, data in results.items():
            if isinstance(data, Exception):
                logger.error(f"Error fetching {source}: {data}")

        # Store snapshots
        await self._store_snapshots(results)

        logger.info(f"Data fetch complete. Matches: {len(results['betfair'])}")
        return results

    async def _wrap_sync(self, func):
        """Wrap synchronous function for async execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    async def _store_snapshots(self, data: dict) -> None:
        """Store fetched data as snapshots in database."""
        with db.session() as session:
            now = datetime.now(timezone.utc)

            # Store team stats
            for team_name, xg_data in data.get("xg", {}).items():
                elo_data = data.get("elo", {}).get(team_name)
                if xg_data:
                    stats = TeamStats(
                        team_name=team_name,
                        snapshot_date=now,
                        xg_for=xg_data.xg_for,
                        xga_against=xg_data.xga_against,
                        xpts=xg_data.xpts,
                        home_xg=xg_data.home_xg,
                        away_xg=xg_data.away_xg,
                        home_xga=xg_data.home_xga,
                        away_xga=xg_data.away_xga,
                        elo_rating=elo_data.elo_rating if elo_data else 0,
                        elo_rank=elo_data.rank if elo_data else 0,
                    )
                    session.add(stats)

            # Store match odds snapshots
            for bf_odds in data.get("betfair", []):
                # Find or create match
                match = session.query(Match).filter_by(
                    external_id=bf_odds.match_id
                ).first()

                if not match:
                    match = Match(
                        external_id=bf_odds.match_id,
                        home_team=bf_odds.home_team,
                        away_team=bf_odds.away_team,
                        kickoff=bf_odds.kickoff,
                    )
                    session.add(match)
                    session.flush()

                # Find matching Polymarket data
                poly_match = self._find_matching_poly(
                    bf_odds, data.get("polymarket", [])
                )

                # Create snapshot
                hours_to_kick = (bf_odds.kickoff - now).total_seconds() / 3600
                snapshot = OddsSnapshot(
                    match_id=match.id,
                    snapshot_time=now,
                    hours_to_kickoff=hours_to_kick,
                    bf_home_odds=bf_odds.home_win_odds,
                    bf_draw_odds=bf_odds.draw_odds,
                    bf_away_odds=bf_odds.away_win_odds,
                    bf_over_2_5_odds=bf_odds.over_2_5_odds,
                    bf_under_2_5_odds=bf_odds.under_2_5_odds,
                    bf_btts_yes_odds=bf_odds.btts_yes_odds,
                    bf_btts_no_odds=bf_odds.btts_no_odds,
                    pm_home_price=poly_match.home_win_price if poly_match else None,
                    pm_draw_price=poly_match.draw_price if poly_match else None,
                    pm_away_price=poly_match.away_win_price if poly_match else None,
                    pm_over_2_5_price=poly_match.over_2_5_price if poly_match else None,
                    pm_btts_yes_price=poly_match.btts_yes_price if poly_match else None,
                    # Handle both PolymarketPrices.home_win_liquidity and DomeApiPrices.liquidity
                    pm_home_liquidity=getattr(poly_match, 'home_win_liquidity', None) or getattr(poly_match, 'liquidity', None) if poly_match else None,
                )
                session.add(snapshot)

    def _find_matching_poly(
        self,
        bf_odds: BetfairOdds,
        poly_data: list[Union[PolymarketPrices, PolymarketMatchPrices]]
    ) -> Optional[Union[PolymarketPrices, PolymarketMatchPrices]]:
        """Find Polymarket/DomeAPI data matching Betfair match."""
        for poly in poly_data:
            # Fuzzy match on team names
            bf_home_lower = bf_odds.home_team.lower()
            poly_home_lower = poly.home_team.lower()

            if (
                bf_home_lower in poly_home_lower or
                poly_home_lower in bf_home_lower
            ):
                return poly
        return None

    async def close(self):
        """Clean up resources."""
        self.betfair.logout()
        await self.polymarket.close()
        if self.dome_api:
            await self.dome_api.close()
