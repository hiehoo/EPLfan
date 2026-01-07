"""Model calculation orchestration."""
import logging
from datetime import datetime, timezone
from typing import Optional

from src.models.poisson_matrix import poisson_calc, PoissonResult, PoissonCalculator
from src.models.market_probabilities import MarketProbabilityCalculator, MatchProbabilities
from src.models.weight_engine import weight_engine, WeightedProbabilities, WeightEngine
from src.models.edge_detector import edge_detector, MatchEdgeAnalysis, EdgeSignal, SignalDirection, EdgeDetector

logger = logging.getLogger(__name__)

__all__ = [
    "MatchAnalyzer",
    "match_analyzer",
    "poisson_calc",
    "PoissonResult",
    "PoissonCalculator",
    "MarketProbabilityCalculator",
    "MatchProbabilities",
    "weight_engine",
    "WeightedProbabilities",
    "WeightEngine",
    "edge_detector",
    "MatchEdgeAnalysis",
    "EdgeSignal",
    "SignalDirection",
    "EdgeDetector",
]


class MatchAnalyzer:
    """Complete match analysis pipeline."""

    def __init__(self):
        self.prob_calculator = MarketProbabilityCalculator()

    def analyze(
        self,
        # Match info
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
        # Betfair odds
        bf_home_odds: float,
        bf_draw_odds: float,
        bf_away_odds: float,
        bf_over_2_5_odds: Optional[float] = None,
        bf_under_2_5_odds: Optional[float] = None,
        bf_btts_yes_odds: Optional[float] = None,
        bf_btts_no_odds: Optional[float] = None,
        # Polymarket prices
        pm_home_price: Optional[float] = None,
        pm_draw_price: Optional[float] = None,
        pm_away_price: Optional[float] = None,
        pm_over_2_5_price: Optional[float] = None,
        pm_under_2_5_price: Optional[float] = None,
        pm_btts_yes_price: Optional[float] = None,
        pm_btts_no_price: Optional[float] = None,
        # Preset overrides
        preset_1x2: Optional[str] = None,
        preset_ou: Optional[str] = None,
        preset_btts: Optional[str] = None,
    ) -> tuple[MatchProbabilities, WeightedProbabilities, MatchEdgeAnalysis]:
        """Run complete analysis pipeline for a match.

        Returns:
            (raw_probabilities, weighted_fair_prices, edge_analysis)
        """
        # Calculate hours to kickoff
        now = datetime.now(timezone.utc)
        if kickoff.tzinfo is None:
            # Assume UTC if no timezone
            kickoff = kickoff.replace(tzinfo=timezone.utc)

        hours_to_kickoff = max(
            0,
            (kickoff - now).total_seconds() / 3600
        )

        # Step 1: Calculate all raw probabilities
        match_probs = self.prob_calculator.calculate_all(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            kickoff=kickoff,
            home_xg=home_xg,
            away_xg=away_xg,
            home_xga=home_xga,
            away_xga=away_xga,
            home_elo=home_elo,
            away_elo=away_elo,
            bf_home_odds=bf_home_odds,
            bf_draw_odds=bf_draw_odds,
            bf_away_odds=bf_away_odds,
            bf_over_2_5_odds=bf_over_2_5_odds,
            bf_under_2_5_odds=bf_under_2_5_odds,
            bf_btts_yes_odds=bf_btts_yes_odds,
            bf_btts_no_odds=bf_btts_no_odds,
            pm_home_price=pm_home_price,
            pm_draw_price=pm_draw_price,
            pm_away_price=pm_away_price,
            pm_over_2_5_price=pm_over_2_5_price,
            pm_under_2_5_price=pm_under_2_5_price,
            pm_btts_yes_price=pm_btts_yes_price,
            pm_btts_no_price=pm_btts_no_price,
        )

        # Step 2: Apply weights to get fair probabilities
        weighted_probs = weight_engine.calculate_all_markets(
            match_probs=match_probs,
            hours_to_kickoff=hours_to_kickoff,
            preset_1x2=preset_1x2,
            preset_ou=preset_ou,
            preset_btts=preset_btts,
        )

        # Step 3: Detect edges
        edge_analysis = edge_detector.analyze_match(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            fair_probs=weighted_probs,
            market_probs=match_probs,
        )

        if edge_analysis.best_signal:
            logger.info(
                f"Analyzed {home_team} vs {away_team}: "
                f"Best edge = {edge_analysis.best_signal.edge_pct:.1f}% on {edge_analysis.best_signal.market_type}"
            )
        else:
            logger.info(f"Analyzed {home_team} vs {away_team}: No actionable edges")

        return match_probs, weighted_probs, edge_analysis


# Singleton instance
match_analyzer = MatchAnalyzer()
