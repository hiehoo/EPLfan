"""Calculate fair probabilities for each market type."""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.models.poisson_matrix import poisson_calc, PoissonResult
from src.fetchers.clubelo_fetcher import ClubELOFetcher

logger = logging.getLogger(__name__)


@dataclass
class MatchProbabilities:
    """All probability calculations for a match."""
    match_id: str
    home_team: str
    away_team: str
    kickoff: datetime

    # Raw Poisson-based probabilities (from xG)
    poisson_result: PoissonResult

    # ELO-based probabilities
    elo_1x2: dict  # {"home": float, "draw": float, "away": float}

    # Betfair implied probabilities
    betfair_1x2: dict
    betfair_ou: dict
    betfair_btts: dict

    # Polymarket prices (already probabilities as decimals)
    polymarket_1x2: dict
    polymarket_ou: dict
    polymarket_btts: dict


class MarketProbabilityCalculator:
    """Calculate probabilities from all data sources."""

    def __init__(self):
        self.elo_calc = ClubELOFetcher()

    def calculate_all(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        # xG data
        home_xg: float,
        away_xg: float,
        home_xga: float,
        away_xga: float,
        # ELO data
        home_elo: float,
        away_elo: float,
        # Betfair odds (decimal)
        bf_home_odds: float,
        bf_draw_odds: float,
        bf_away_odds: float,
        bf_over_2_5_odds: Optional[float] = None,
        bf_under_2_5_odds: Optional[float] = None,
        bf_btts_yes_odds: Optional[float] = None,
        bf_btts_no_odds: Optional[float] = None,
        # Polymarket prices (0-1)
        pm_home_price: Optional[float] = None,
        pm_draw_price: Optional[float] = None,
        pm_away_price: Optional[float] = None,
        pm_over_2_5_price: Optional[float] = None,
        pm_under_2_5_price: Optional[float] = None,
        pm_btts_yes_price: Optional[float] = None,
        pm_btts_no_price: Optional[float] = None,
    ) -> MatchProbabilities:
        """Calculate all probabilities for a match."""

        # 1. Poisson from xG (use match-specific xG if available, else season avg)
        # For home team: use their home xG, opponent's away xGA
        effective_home_xg = (home_xg + away_xga) / 2
        effective_away_xg = (away_xg + home_xga) / 2

        poisson_result = poisson_calc.calculate(effective_home_xg, effective_away_xg)

        # 2. ELO-based 1X2
        elo_1x2 = self.elo_calc.calculate_win_probability(home_elo, away_elo)

        # 3. Betfair implied probabilities (remove overround)
        betfair_1x2 = self._odds_to_probabilities({
            "home": bf_home_odds,
            "draw": bf_draw_odds,
            "away": bf_away_odds,
        })

        betfair_ou = {}
        if bf_over_2_5_odds and bf_under_2_5_odds:
            betfair_ou["2.5"] = self._odds_to_probabilities({
                "over": bf_over_2_5_odds,
                "under": bf_under_2_5_odds,
            })

        betfair_btts = {}
        if bf_btts_yes_odds and bf_btts_no_odds:
            betfair_btts = self._odds_to_probabilities({
                "yes": bf_btts_yes_odds,
                "no": bf_btts_no_odds,
            })

        # 4. Polymarket prices (already probabilities, just structure them)
        polymarket_1x2 = {
            "home": pm_home_price or 0,
            "draw": pm_draw_price or 0,
            "away": pm_away_price or 0,
        }

        polymarket_ou = {}
        if pm_over_2_5_price:
            polymarket_ou["2.5"] = {
                "over": pm_over_2_5_price,
                "under": pm_under_2_5_price or (1 - pm_over_2_5_price),
            }

        polymarket_btts = {
            "yes": pm_btts_yes_price or 0,
            "no": 1 - (pm_btts_yes_price or 0) if pm_btts_yes_price else 0,
        }

        return MatchProbabilities(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            kickoff=kickoff,
            poisson_result=poisson_result,
            elo_1x2=elo_1x2,
            betfair_1x2=betfair_1x2,
            betfair_ou=betfair_ou,
            betfair_btts=betfair_btts,
            polymarket_1x2=polymarket_1x2,
            polymarket_ou=polymarket_ou,
            polymarket_btts=polymarket_btts,
        )

    def _odds_to_probabilities(self, odds_dict: dict) -> dict:
        """Convert decimal odds to probabilities, removing overround.

        Example: odds {home: 2.0, draw: 3.5, away: 4.0}
        Implied probs: {home: 0.5, draw: 0.286, away: 0.25} = 1.036 (3.6% overround)
        Normalized: {home: 0.483, draw: 0.276, away: 0.241} = 1.0
        """
        implied = {k: 1 / v for k, v in odds_dict.items() if v and v > 0}

        total = sum(implied.values())
        if total == 0:
            return {k: 0 for k in odds_dict}

        # Normalize to remove overround
        return {k: v / total for k, v in implied.items()}
