"""Fetch odds from Betfair Exchange for EPL matches."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import betfairlightweight
from betfairlightweight.filters import market_filter, time_range

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BetfairOdds:
    """Betfair odds for a match."""
    match_id: str
    home_team: str
    away_team: str
    kickoff: datetime

    # 1X2 odds (decimal)
    home_win_odds: float
    draw_odds: float
    away_win_odds: float

    # Over/Under odds (decimal)
    over_1_5_odds: Optional[float] = None
    under_1_5_odds: Optional[float] = None
    over_2_5_odds: Optional[float] = None
    under_2_5_odds: Optional[float] = None
    over_3_5_odds: Optional[float] = None
    under_3_5_odds: Optional[float] = None

    # BTTS odds (decimal)
    btts_yes_odds: Optional[float] = None
    btts_no_odds: Optional[float] = None

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BetfairFetcher:
    """Fetch EPL odds from Betfair Exchange."""

    EPL_COMPETITION_ID = "10932509"  # English Premier League

    def __init__(self):
        if not settings.betfair.is_configured():
            logger.warning("Betfair credentials not configured - fetcher will not work")

        self.client = betfairlightweight.APIClient(
            username=settings.betfair.username,
            password=settings.betfair.password.get_secret_value(),
            app_key=settings.betfair.app_key.get_secret_value(),
            certs=str(settings.betfair.certs_path),
        )
        self._logged_in = False

    def login(self) -> bool:
        """Login to Betfair API.

        Returns:
            True if login successful, False otherwise
        """
        if self._logged_in:
            return True

        if not settings.betfair.is_configured():
            logger.warning("Betfair credentials not configured")
            return False

        try:
            self.client.login()
            self._logged_in = True
            logger.info("Logged in to Betfair")
            return True
        except Exception as e:
            logger.error(f"Betfair login failed: {e}")
            self._logged_in = False
            return False

    def logout(self) -> None:
        """Logout from Betfair API."""
        if self._logged_in:
            self.client.logout()
            self._logged_in = False
            logger.info("Logged out from Betfair")

    def fetch_upcoming_matches(
        self,
        hours_ahead: int = 48
    ) -> list[BetfairOdds]:
        """Fetch odds for upcoming EPL matches."""
        self.login()

        now = datetime.now(timezone.utc)
        time_filter = time_range(
            from_=now,
            to=now + timedelta(hours=hours_ahead),
        )

        # Get Match Odds markets (1X2)
        match_odds_markets = self.client.betting.list_market_catalogue(
            filter=market_filter(
                competition_ids=[self.EPL_COMPETITION_ID],
                market_type_codes=["MATCH_ODDS"],
                market_start_time=time_filter,
            ),
            market_projection=["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
            max_results=50,
        )

        results = []
        for market in match_odds_markets:
            odds = self._fetch_market_odds(market)
            if odds:
                # Fetch O/U and BTTS for same event
                self._enrich_with_ou_btts(odds, market.event.id)
                results.append(odds)

        logger.info(f"Fetched {len(results)} EPL matches from Betfair")
        return results

    def _fetch_market_odds(self, market) -> Optional[BetfairOdds]:
        """Extract odds from market catalogue."""
        try:
            # Get price data
            price_data = self.client.betting.list_market_book(
                market_ids=[market.market_id],
                price_projection={"priceData": ["EX_BEST_OFFERS"]},
            )

            if not price_data or not price_data[0].runners:
                return None

            runners = {r.selection_id: r for r in price_data[0].runners}
            runner_names = {r.selection_id: r.runner_name for r in market.runners}

            # Map runners to home/draw/away
            home_odds = draw_odds = away_odds = None
            event_name = market.event.name if market.event else ""

            for sel_id, runner in runners.items():
                name = runner_names.get(sel_id, "").lower()
                best_back = (
                    runner.ex.available_to_back[0].price
                    if runner.ex.available_to_back and len(runner.ex.available_to_back) > 0
                    else None
                )

                if best_back:
                    if "draw" in name or name == "the draw":
                        draw_odds = best_back
                    elif " v " in event_name:
                        home_team = event_name.split(" v ")[0].strip().lower()
                        if home_team in name or name in home_team:
                            home_odds = best_back
                        else:
                            away_odds = best_back
                    else:
                        # Fallback: first non-draw is home, second is away
                        if home_odds is None:
                            home_odds = best_back
                        else:
                            away_odds = best_back

            if not all([home_odds, draw_odds, away_odds]):
                return None

            # Parse teams from event name
            teams = event_name.split(" v ") if " v " in event_name else ["", ""]
            home_team = teams[0].strip() if len(teams) > 0 else ""
            away_team = teams[1].strip() if len(teams) > 1 else ""

            return BetfairOdds(
                match_id=market.event.id,
                home_team=home_team,
                away_team=away_team,
                kickoff=market.market_start_time,
                home_win_odds=home_odds,
                draw_odds=draw_odds,
                away_win_odds=away_odds,
                fetched_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Error fetching odds for {market.event.name if market.event else 'unknown'}: {e}")
            return None

    def _enrich_with_ou_btts(self, odds: BetfairOdds, event_id: str) -> None:
        """Add O/U and BTTS odds to existing BetfairOdds object."""
        try:
            # Fetch Over/Under 2.5 Goals market
            ou_markets = self.client.betting.list_market_catalogue(
                filter=market_filter(
                    event_ids=[event_id],
                    market_type_codes=["OVER_UNDER_25"],
                ),
                market_projection=["RUNNER_DESCRIPTION"],
                max_results=1,
            )

            if ou_markets:
                ou_prices = self.client.betting.list_market_book(
                    market_ids=[ou_markets[0].market_id],
                    price_projection={"priceData": ["EX_BEST_OFFERS"]},
                )
                if ou_prices and ou_prices[0].runners:
                    for runner in ou_prices[0].runners:
                        best_back = (
                            runner.ex.available_to_back[0].price
                            if runner.ex.available_to_back and len(runner.ex.available_to_back) > 0
                            else None
                        )
                        if best_back:
                            # Determine if Over or Under based on selection name
                            runner_name = next(
                                (r.runner_name for r in ou_markets[0].runners if r.selection_id == runner.selection_id),
                                ""
                            )
                            if "over" in runner_name.lower():
                                odds.over_2_5_odds = best_back
                            else:
                                odds.under_2_5_odds = best_back

            # Fetch BTTS market
            btts_markets = self.client.betting.list_market_catalogue(
                filter=market_filter(
                    event_ids=[event_id],
                    market_type_codes=["BOTH_TEAMS_TO_SCORE"],
                ),
                market_projection=["RUNNER_DESCRIPTION"],
                max_results=1,
            )

            if btts_markets:
                btts_prices = self.client.betting.list_market_book(
                    market_ids=[btts_markets[0].market_id],
                    price_projection={"priceData": ["EX_BEST_OFFERS"]},
                )
                if btts_prices and btts_prices[0].runners:
                    for runner in btts_prices[0].runners:
                        best_back = (
                            runner.ex.available_to_back[0].price
                            if runner.ex.available_to_back and len(runner.ex.available_to_back) > 0
                            else None
                        )
                        if best_back:
                            runner_name = next(
                                (r.runner_name for r in btts_markets[0].runners if r.selection_id == runner.selection_id),
                                ""
                            )
                            if "yes" in runner_name.lower():
                                odds.btts_yes_odds = best_back
                            else:
                                odds.btts_no_odds = best_back

        except Exception as e:
            logger.warning(f"Error enriching O/U/BTTS for event {event_id}: {e}")

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()
