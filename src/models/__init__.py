"""Model calculation orchestration."""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.models.poisson_matrix import poisson_calc, PoissonResult, PoissonCalculator
from src.models.market_probabilities import MarketProbabilityCalculator, MatchProbabilities
from src.models.weight_engine import weight_engine, WeightedProbabilities, WeightEngine
from src.models.edge_detector import edge_detector, MatchEdgeAnalysis, EdgeSignal, SignalDirection, EdgeDetector, HybridEdgeDetector
from src.models.feature_builder import MatchFeatures

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

    def __init__(self, use_ml: bool = True):
        self.prob_calculator = MarketProbabilityCalculator()
        self.hybrid_detector = HybridEdgeDetector(use_ml=use_ml)

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
        # ML feature stats (optional)
        home_shots_pg: Optional[float] = None,
        away_shots_pg: Optional[float] = None,
        home_volatility: Optional[float] = None,
        away_volatility: Optional[float] = None,
        home_conversion: Optional[float] = None,
        away_conversion: Optional[float] = None,
        is_elite_match: bool = False,
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

        # Step 3: Build ML features if stats available
        features = None
        if home_shots_pg is not None and away_shots_pg is not None:
            shot_ratio = home_shots_pg / away_shots_pg if away_shots_pg > 0 else 1.0
            xg_diff = home_xg - away_xga
            combined_volatility = (
                (home_volatility or 0) + (away_volatility or 0)
            ) / 2

            # Build feature array (must match MatchFeatures.to_array order)
            features = np.array([
                home_shots_pg,
                away_shots_pg,
                shot_ratio,
                home_xg,
                away_xg,
                xg_diff,
                combined_volatility,
                home_xg,  # home_form_xg (use xg as fallback)
                away_xg,  # away_form_xg
                home_conversion or 0.0,
                away_conversion or 0.0,
                home_elo - away_elo,  # elo_diff
                1 / bf_over_2_5_odds if bf_over_2_5_odds else 0.5,  # bf_over_implied
                1.0 if is_elite_match else 0.0,
            ])

        # Step 4: Detect edges (use hybrid if features available)
        if features is not None and self.hybrid_detector.use_ml:
            edge_analysis = self.hybrid_detector.analyze_match_hybrid(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                fair_probs=weighted_probs,
                market_probs=match_probs,
                features=features,
            )
        else:
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
