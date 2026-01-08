"""Detect edges between model fair price and market price."""
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.config.settings import settings
from src.models.base_rate_tracker import base_rate_tracker

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Source of probability estimate."""
    POISSON = "poisson"
    ML_CLASSIFIER = "ml"
    HYBRID = "hybrid"


class SignalDirection(Enum):
    """Trading signal direction."""
    NONE = "none"
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"
    YES = "yes"
    NO = "no"


@dataclass
class EdgeSignal:
    """Edge detection result for a single market."""
    market_type: str  # "1x2", "ou", "btts"
    outcome: str  # "home", "over", "yes", etc.
    fair_prob: float
    market_prob: float
    edge: float  # fair - market (positive = underpriced)
    edge_pct: float  # edge as percentage
    direction: SignalDirection
    is_actionable: bool  # Edge exceeds threshold
    threshold_used: float
    # Base rate fields for true edge calculation
    base_rate: float = 0.5
    true_edge: float = 0.0  # edge vs base rate
    skill_component: float = 0.0  # model_prob - base_rate
    is_elite_match: bool = False


@dataclass
class HybridEdgeSignal(EdgeSignal):
    """Edge signal with hybrid model info."""
    model_source: ModelSource = ModelSource.POISSON
    poisson_prob: float = 0.0
    ml_prob: Optional[float] = None
    ml_confidence: Optional[str] = None
    agreement_score: float = 1.0  # How much Poisson and ML agree


@dataclass
class MatchEdgeAnalysis:
    """Complete edge analysis for a match."""
    match_id: str
    home_team: str
    away_team: str

    # Individual market signals
    signal_1x2: Optional[EdgeSignal]
    signal_ou: Optional[EdgeSignal]
    signal_btts: Optional[EdgeSignal]

    # Best overall signal
    best_signal: Optional[EdgeSignal]

    # All raw edges for display
    all_edges: dict  # {"1x2": {"home": edge, ...}, "ou": {...}, "btts": {...}}


class EdgeDetector:
    """Detect trading edges across all markets."""

    def __init__(self):
        self.threshold_1x2 = settings.edge_threshold_1x2
        self.threshold_ou = settings.edge_threshold_ou
        self.threshold_btts = settings.edge_threshold_btts

    def _validate_edge(self, edge: float) -> float:
        """Validate edge value, return 0.0 if invalid."""
        if edge is None or math.isnan(edge) or math.isinf(edge):
            logger.warning(f"Invalid edge value: {edge}, defaulting to 0.0")
            return 0.0
        return edge

    def analyze_match(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        fair_probs,  # WeightedProbabilities
        market_probs,  # MatchProbabilities
    ) -> MatchEdgeAnalysis:
        """Analyze edges for all markets in a match."""

        # Calculate all edges
        all_edges = self._calculate_all_edges(fair_probs, market_probs)

        # Find best signal per market
        signal_1x2 = self._find_best_edge_in_market(
            all_edges.get("1x2", {}),
            market_type="1x2",
            threshold=self.threshold_1x2,
            home_team=home_team,
            away_team=away_team,
        )

        signal_ou = self._find_best_edge_in_market(
            all_edges.get("ou", {}),
            market_type="ou",
            threshold=self.threshold_ou,
            home_team=home_team,
            away_team=away_team,
        )

        signal_btts = self._find_best_edge_in_market(
            all_edges.get("btts", {}),
            market_type="btts",
            threshold=self.threshold_btts,
            home_team=home_team,
            away_team=away_team,
        )

        # Find overall best signal
        actionable_signals = [s for s in [signal_1x2, signal_ou, signal_btts] if s and s.is_actionable]
        best_signal = max(actionable_signals, key=lambda s: abs(s.edge), default=None)

        return MatchEdgeAnalysis(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            signal_1x2=signal_1x2,
            signal_ou=signal_ou,
            signal_btts=signal_btts,
            best_signal=best_signal,
            all_edges=all_edges,
        )

    def _calculate_all_edges(self, fair_probs, market_probs) -> dict:
        """Calculate edges for all outcomes in all markets."""
        edges = {}

        # 1X2 edges
        edges["1x2"] = {}
        for outcome in ["home", "draw", "away"]:
            fair = fair_probs.fair_1x2.get(outcome, 0)
            market = market_probs.polymarket_1x2.get(outcome, 0)
            if market > 0 and fair is not None and not (isinstance(fair, float) and math.isnan(fair)):
                edges["1x2"][outcome] = {
                    "fair": fair,
                    "market": market,
                    "edge": self._validate_edge(fair - market),
                }

        # O/U edges
        edges["ou"] = {}
        fair_ou = fair_probs.fair_ou.get("2.5", {})
        market_ou = market_probs.polymarket_ou.get("2.5", {})
        for outcome in ["over", "under"]:
            fair = fair_ou.get(outcome, 0)
            market = market_ou.get(outcome, 0)
            if market > 0 and fair is not None and not (isinstance(fair, float) and math.isnan(fair)):
                edges["ou"][outcome] = {
                    "fair": fair,
                    "market": market,
                    "edge": self._validate_edge(fair - market),
                }

        # BTTS edges
        edges["btts"] = {}
        for outcome in ["yes", "no"]:
            fair = fair_probs.fair_btts.get(outcome, 0)
            market = market_probs.polymarket_btts.get(outcome, 0)
            if market > 0 and fair is not None and not (isinstance(fair, float) and math.isnan(fair)):
                edges["btts"][outcome] = {
                    "fair": fair,
                    "market": market,
                    "edge": self._validate_edge(fair - market),
                }

        return edges

    def _find_best_edge_in_market(
        self,
        market_edges: dict,
        market_type: str,
        threshold: float,
        home_team: str = "",
        away_team: str = "",
    ) -> Optional[EdgeSignal]:
        """Find the best edge within a single market."""
        if not market_edges:
            return None

        # Find outcome with largest TRUE edge (not naive)
        best_outcome = None
        best_true_edge = 0
        best_data = None

        for outcome, edge_data in market_edges.items():
            true_edge, skill, base_rate, is_elite = self.calculate_true_edge(
                model_prob=edge_data["fair"],
                market_prob=edge_data["market"],
                market_type=market_type,
                outcome=outcome,
                home_team=home_team,
                away_team=away_team,
            )

            if abs(true_edge) > abs(best_true_edge):
                best_true_edge = true_edge
                best_outcome = outcome
                best_data = {
                    **edge_data,
                    "true_edge": true_edge,
                    "skill_component": skill,
                    "base_rate": base_rate,
                    "is_elite": is_elite,
                }

        if best_outcome is None:
            return None

        direction = self._get_direction(market_type, best_outcome, best_true_edge)

        return EdgeSignal(
            market_type=market_type,
            outcome=best_outcome,
            fair_prob=best_data["fair"],
            market_prob=best_data["market"],
            edge=best_data["fair"] - best_data["market"],  # Keep naive for display
            edge_pct=(best_data["fair"] - best_data["market"]) * 100,
            direction=direction,
            is_actionable=abs(best_true_edge) >= max(threshold, 0.01),  # Use TRUE edge
            threshold_used=threshold,
            base_rate=best_data["base_rate"],
            true_edge=best_true_edge,
            skill_component=best_data["skill_component"],
            is_elite_match=best_data["is_elite"],
        )

    def calculate_true_edge(
        self,
        model_prob: float,
        market_prob: float,
        market_type: str,
        outcome: str,
        line: Optional[float] = None,
        home_team: str = "",
        away_team: str = "",
    ) -> tuple[float, float, float, bool]:
        """Calculate true edge accounting for base rate.

        Returns:
            (true_edge, skill_component, base_rate, is_elite)
        """
        base_rate = base_rate_tracker.get_base_rate(market_type, outcome, line)

        # Naive edge (what we had before)
        naive_edge = self._validate_edge(model_prob - market_prob)

        # Skill component: how much better than just guessing?
        # If base_rate > 0.5, naive strategy is to always predict that outcome
        skill_component = self._validate_edge(model_prob - max(base_rate, 1 - base_rate))

        # True edge: minimum of naive and skill
        # This prevents false confidence from high base rate outcomes
        true_edge = self._validate_edge(
            min(naive_edge, skill_component) if skill_component > 0 else 0
        )

        # Elite team boost
        is_elite = (
            base_rate_tracker.is_elite_team(home_team) or
            base_rate_tracker.is_elite_team(away_team)
        )

        if is_elite and true_edge > 0:
            adjustment = base_rate_tracker.get_elite_adjustment(home_team)
            if not base_rate_tracker.is_elite_team(home_team):
                adjustment = base_rate_tracker.get_elite_adjustment(away_team)
            true_edge *= adjustment

        return true_edge, skill_component, base_rate, is_elite

    def _get_direction(self, market_type: str, outcome: str, edge: float) -> SignalDirection:
        """Determine signal direction."""
        if edge > 0:  # Underpriced - buy
            direction_map = {
                "home": SignalDirection.HOME,
                "draw": SignalDirection.DRAW,
                "away": SignalDirection.AWAY,
                "over": SignalDirection.OVER,
                "under": SignalDirection.UNDER,
                "yes": SignalDirection.YES,
                "no": SignalDirection.NO,
            }
            return direction_map.get(outcome, SignalDirection.NONE)
        else:  # Overpriced - signal the opposite
            opposites = {
                "home": SignalDirection.AWAY,
                "away": SignalDirection.HOME,
                "draw": SignalDirection.DRAW,  # Draw stays draw
                "over": SignalDirection.UNDER,
                "under": SignalDirection.OVER,
                "yes": SignalDirection.NO,
                "no": SignalDirection.YES,
            }
            return opposites.get(outcome, SignalDirection.NONE)


class HybridEdgeDetector(EdgeDetector):
    """Edge detector combining Poisson and ML predictions."""

    # Weight for ML vs Poisson
    ML_WEIGHT_HIGH_CONFIDENCE = 0.6
    ML_WEIGHT_MEDIUM_CONFIDENCE = 0.4
    ML_WEIGHT_LOW_CONFIDENCE = 0.2

    def __init__(self, use_ml: bool = True):
        super().__init__()
        self._use_ml = use_ml
        self._ml_initialized = False

    @property
    def use_ml(self) -> bool:
        """Check if ML is available and enabled."""
        if not self._use_ml:
            return False
        if not self._ml_initialized:
            try:
                from src.models.ml_classifier import ml_classifier_ou
                self._ml_initialized = ml_classifier_ou.is_trained
            except ImportError:
                self._ml_initialized = False
        return self._ml_initialized

    def analyze_match_hybrid(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        fair_probs,  # WeightedProbabilities (Poisson-based)
        market_probs,  # MatchProbabilities
        features=None,  # Optional[MatchFeatures]
    ) -> MatchEdgeAnalysis:
        """Analyze edges using hybrid Poisson + ML approach."""
        from src.models.ml_classifier import ml_classifier_ou, ml_classifier_btts

        # Get base Poisson analysis
        base_analysis = self.analyze_match(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            fair_probs=fair_probs,
            market_probs=market_probs,
        )

        if not self.use_ml or features is None:
            return base_analysis

        # Validate and get ML predictions
        if hasattr(features, 'to_array'):
            feature_array = features.to_array()
        elif isinstance(features, np.ndarray):
            feature_array = features
        else:
            logger.warning(f"Invalid features type: {type(features)}, using base analysis")
            return base_analysis

        ml_ou = ml_classifier_ou.predict_proba(feature_array)
        ml_btts = ml_classifier_btts.predict_proba(feature_array)

        # Enhance O/U signal with ML
        if base_analysis.signal_ou:
            enhanced_ou = self._enhance_with_ml(
                base_signal=base_analysis.signal_ou,
                ml_result=ml_ou,
                market_type="ou",
                home_team=home_team,
                away_team=away_team,
            )
            base_analysis.signal_ou = enhanced_ou

        # Enhance BTTS signal with ML
        if base_analysis.signal_btts:
            enhanced_btts = self._enhance_with_ml(
                base_signal=base_analysis.signal_btts,
                ml_result=ml_btts,
                market_type="btts",
                home_team=home_team,
                away_team=away_team,
            )
            base_analysis.signal_btts = enhanced_btts

        # Re-evaluate best signal
        actionable = [
            s for s in [base_analysis.signal_1x2, base_analysis.signal_ou, base_analysis.signal_btts]
            if s and s.is_actionable
        ]
        base_analysis.best_signal = max(actionable, key=lambda s: abs(s.true_edge), default=None)

        return base_analysis

    def _enhance_with_ml(
        self,
        base_signal: EdgeSignal,
        ml_result,
        market_type: str,
        home_team: str = "",
        away_team: str = "",
    ) -> HybridEdgeSignal:
        """Enhance Poisson signal with ML prediction."""

        # Calculate ML weight based on confidence
        ml_weight = {
            "high": self.ML_WEIGHT_HIGH_CONFIDENCE,
            "medium": self.ML_WEIGHT_MEDIUM_CONFIDENCE,
            "low": self.ML_WEIGHT_LOW_CONFIDENCE,
        }.get(ml_result.confidence, 0.2)

        poisson_weight = 1 - ml_weight

        # Determine ML prob for this outcome
        if base_signal.outcome in ("over", "yes"):
            ml_prob = ml_result.probability
        elif base_signal.outcome in ("under", "no"):
            ml_prob = 1 - ml_result.probability
        else:
            ml_prob = base_signal.fair_prob

        # Blend probabilities
        hybrid_prob = (
            poisson_weight * base_signal.fair_prob +
            ml_weight * ml_prob
        )

        # Calculate agreement score (0-1)
        agreement = 1 - abs(base_signal.fair_prob - ml_prob)

        # Recalculate true edge with hybrid prob
        true_edge, skill, base_rate, is_elite = self.calculate_true_edge(
            model_prob=hybrid_prob,
            market_prob=base_signal.market_prob,
            market_type=market_type,
            outcome=base_signal.outcome,
            home_team=home_team,
            away_team=away_team,
        )

        return HybridEdgeSignal(
            market_type=base_signal.market_type,
            outcome=base_signal.outcome,
            fair_prob=hybrid_prob,
            market_prob=base_signal.market_prob,
            edge=hybrid_prob - base_signal.market_prob,
            edge_pct=(hybrid_prob - base_signal.market_prob) * 100,
            direction=base_signal.direction,
            is_actionable=abs(true_edge) >= base_signal.threshold_used,
            threshold_used=base_signal.threshold_used,
            base_rate=base_rate,
            true_edge=true_edge,
            skill_component=skill,
            is_elite_match=is_elite,
            model_source=ModelSource.HYBRID,
            poisson_prob=base_signal.fair_prob,
            ml_prob=ml_prob,
            ml_confidence=ml_result.confidence,
            agreement_score=agreement,
        )


# Singleton instances
edge_detector = EdgeDetector()
hybrid_edge_detector = HybridEdgeDetector()
