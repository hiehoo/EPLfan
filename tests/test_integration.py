"""Integration tests for Phase 4: hybrid edge detection, A/B testing, performance logging."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.models.edge_detector import (
    EdgeDetector,
    EdgeSignal,
    HybridEdgeSignal,
    HybridEdgeDetector,
    ModelSource,
    SignalDirection,
    MatchEdgeAnalysis,
)
from src.models.ml_classifier import ClassificationResult


class TestHybridEdgeDetector:
    """Tests for hybrid edge detection."""

    def test_hybrid_detector_init(self):
        """HybridEdgeDetector should initialize correctly."""
        detector = HybridEdgeDetector(use_ml=False)
        assert detector._use_ml is False
        assert detector.threshold_1x2 > 0

    def test_hybrid_detector_inherits_base(self):
        """HybridEdgeDetector should inherit from EdgeDetector."""
        detector = HybridEdgeDetector()
        assert isinstance(detector, EdgeDetector)

    def test_ml_weight_constants(self):
        """ML weight constants should be defined."""
        detector = HybridEdgeDetector()
        assert detector.ML_WEIGHT_HIGH_CONFIDENCE == 0.6
        assert detector.ML_WEIGHT_MEDIUM_CONFIDENCE == 0.4
        assert detector.ML_WEIGHT_LOW_CONFIDENCE == 0.2

    def test_use_ml_property_when_disabled(self):
        """use_ml should return False when disabled."""
        detector = HybridEdgeDetector(use_ml=False)
        assert detector.use_ml is False

    def test_hybrid_probability_blending_high_confidence(self):
        """High confidence ML should have 60% weight."""
        detector = HybridEdgeDetector(use_ml=False)

        # Manual calculation
        poisson_prob = 0.58
        ml_prob = 0.65
        ml_weight = detector.ML_WEIGHT_HIGH_CONFIDENCE  # 0.6
        poisson_weight = 1 - ml_weight  # 0.4

        expected = poisson_weight * poisson_prob + ml_weight * ml_prob
        # 0.4 * 0.58 + 0.6 * 0.65 = 0.232 + 0.39 = 0.622

        assert abs(expected - 0.622) < 0.01

    def test_hybrid_probability_blending_low_confidence(self):
        """Low confidence ML should have 20% weight."""
        detector = HybridEdgeDetector(use_ml=False)

        poisson_prob = 0.58
        ml_prob = 0.65
        ml_weight = detector.ML_WEIGHT_LOW_CONFIDENCE  # 0.2
        poisson_weight = 1 - ml_weight  # 0.8

        expected = poisson_weight * poisson_prob + ml_weight * ml_prob
        # 0.8 * 0.58 + 0.2 * 0.65 = 0.464 + 0.13 = 0.594

        assert abs(expected - 0.594) < 0.01


class TestHybridEdgeSignal:
    """Tests for HybridEdgeSignal dataclass."""

    def test_hybrid_signal_inherits_edge_signal(self):
        """HybridEdgeSignal should inherit all EdgeSignal fields."""
        signal = HybridEdgeSignal(
            market_type="ou",
            outcome="over",
            fair_prob=0.62,
            market_prob=0.50,
            edge=0.12,
            edge_pct=12.0,
            direction=SignalDirection.OVER,
            is_actionable=True,
            threshold_used=0.07,
        )

        assert signal.market_type == "ou"
        assert signal.fair_prob == 0.62
        assert signal.is_actionable is True

    def test_hybrid_signal_has_model_source(self):
        """HybridEdgeSignal should have model source field."""
        signal = HybridEdgeSignal(
            market_type="ou",
            outcome="over",
            fair_prob=0.62,
            market_prob=0.50,
            edge=0.12,
            edge_pct=12.0,
            direction=SignalDirection.OVER,
            is_actionable=True,
            threshold_used=0.07,
            model_source=ModelSource.HYBRID,
        )

        assert signal.model_source == ModelSource.HYBRID

    def test_hybrid_signal_ml_fields(self):
        """HybridEdgeSignal should have ML-specific fields."""
        signal = HybridEdgeSignal(
            market_type="ou",
            outcome="over",
            fair_prob=0.62,
            market_prob=0.50,
            edge=0.12,
            edge_pct=12.0,
            direction=SignalDirection.OVER,
            is_actionable=True,
            threshold_used=0.07,
            model_source=ModelSource.HYBRID,
            poisson_prob=0.58,
            ml_prob=0.65,
            ml_confidence="high",
            agreement_score=0.93,
        )

        assert signal.poisson_prob == 0.58
        assert signal.ml_prob == 0.65
        assert signal.ml_confidence == "high"
        assert signal.agreement_score == 0.93

    def test_agreement_score_calculation(self):
        """Agreement score should reflect model similarity."""
        # High agreement (both predict ~60%)
        poisson = 0.58
        ml = 0.64
        agreement = 1 - abs(poisson - ml)
        assert agreement == pytest.approx(0.94, abs=0.01)

        # Low agreement (Poisson 40%, ML 70%)
        poisson_low = 0.40
        ml_high = 0.70
        agreement_low = 1 - abs(poisson_low - ml_high)
        assert agreement_low == pytest.approx(0.70, abs=0.01)


class TestModelSource:
    """Tests for ModelSource enum."""

    def test_model_source_values(self):
        """ModelSource should have correct values."""
        assert ModelSource.POISSON.value == "poisson"
        assert ModelSource.ML_CLASSIFIER.value == "ml"
        assert ModelSource.HYBRID.value == "hybrid"


class TestABTestingFramework:
    """Tests for A/B testing framework."""

    def test_ab_test_result_creation(self):
        """ABTestResult should store prediction data."""
        from src.models.ab_testing import ABTestResult

        result = ABTestResult(
            match_id="123",
            market_type="ou",
            outcome="over",
            poisson_prob=0.58,
            ml_prob=0.65,
            hybrid_prob=0.62,
            market_prob=0.50,
        )

        assert result.match_id == "123"
        assert result.poisson_prob == 0.58
        assert result.ml_prob == 0.65

    def test_ab_test_result_edges(self):
        """ABTestResult should calculate edges correctly."""
        from src.models.ab_testing import ABTestResult

        result = ABTestResult(
            match_id="123",
            market_type="ou",
            outcome="over",
            poisson_prob=0.58,
            ml_prob=0.65,
            hybrid_prob=0.62,
            market_prob=0.50,
            poisson_edge=0.08,
            ml_edge=0.15,
            hybrid_edge=0.12,
        )

        assert result.poisson_edge == 0.08
        assert result.ml_edge == 0.15
        assert result.hybrid_edge == 0.12

    def test_ab_tracker_log_prediction(self):
        """ABTestTracker should log predictions."""
        from src.models.ab_testing import ABTestTracker

        tracker = ABTestTracker()
        tracker.results = []  # Reset

        tracker.log_prediction(
            match_id="123",
            market_type="ou",
            outcome="over",
            poisson_prob=0.58,
            ml_prob=0.65,
            hybrid_prob=0.62,
            market_prob=0.50,
        )

        assert len(tracker.results) == 1
        assert tracker.results[0].poisson_edge == pytest.approx(0.08, abs=0.001)
        assert tracker.results[0].ml_edge == pytest.approx(0.15, abs=0.001)

    def test_ab_tracker_summary_empty(self):
        """ABTestTracker summary should handle empty results."""
        from src.models.ab_testing import ABTestTracker

        tracker = ABTestTracker()
        tracker.results = []

        summary = tracker.get_summary()

        assert summary.n_predictions == 0
        assert summary.poisson_win_rate == 0

    def test_ab_tracker_summary_with_results(self):
        """ABTestTracker should calculate summary stats."""
        from src.models.ab_testing import ABTestTracker, ABTestResult

        tracker = ABTestTracker()
        tracker.results = [
            ABTestResult(
                match_id="1",
                market_type="ou",
                outcome="over",
                poisson_prob=0.6,
                ml_prob=0.7,
                hybrid_prob=0.65,
                market_prob=0.5,
                actual_outcome="over",
                is_correct_poisson=True,
                is_correct_ml=True,
                is_correct_hybrid=True,
                poisson_edge=0.1,
                ml_edge=0.2,
                hybrid_edge=0.15,
            ),
            ABTestResult(
                match_id="2",
                market_type="ou",
                outcome="over",
                poisson_prob=0.55,
                ml_prob=0.45,
                hybrid_prob=0.50,
                market_prob=0.5,
                actual_outcome="under",
                is_correct_poisson=False,
                is_correct_ml=True,  # ML predicted under (prob < 0.5)
                is_correct_hybrid=False,
                poisson_edge=0.05,
                ml_edge=-0.05,
                hybrid_edge=0.0,
            ),
        ]

        summary = tracker.get_summary()

        assert summary.n_predictions == 2
        assert summary.poisson_win_rate == 0.5
        assert summary.ml_win_rate == 1.0  # Both ML predictions correct


class TestPerformanceLogger:
    """Tests for performance logging."""

    def test_performance_logger_init(self, tmp_path):
        """PerformanceLogger should create log directory."""
        from src.models.performance_logger import PerformanceLogger

        logger = PerformanceLogger()
        logger.LOG_DIR = tmp_path / "perf_logs"
        logger.LOG_DIR.mkdir(parents=True, exist_ok=True)

        assert logger.LOG_DIR.exists()

    def test_log_prediction(self, tmp_path):
        """Should log predictions to file."""
        from src.models.performance_logger import PerformanceLogger
        import json

        logger = PerformanceLogger()
        logger.LOG_DIR = tmp_path

        logger.log_prediction(
            match_id="123",
            market_type="ou",
            poisson_prob=0.58,
            ml_prob=0.65,
            hybrid_prob=0.62,
            market_prob=0.50,
            true_edge=0.05,
            is_actionable=True,
        )

        # Check file exists
        files = list(tmp_path.glob("predictions_*.jsonl"))
        assert len(files) == 1

        # Check content
        with open(files[0]) as f:
            data = json.loads(f.readline())
            assert data["match_id"] == "123"
            assert data["true_edge"] == 0.05
            assert data["is_actionable"] is True

    def test_get_daily_stats_empty(self, tmp_path):
        """Should return empty stats for missing file."""
        from src.models.performance_logger import PerformanceLogger

        logger = PerformanceLogger()
        logger.LOG_DIR = tmp_path

        stats = logger.get_daily_stats("20260101")

        assert stats["n_predictions"] == 0

    def test_get_daily_stats(self, tmp_path):
        """Should calculate daily statistics."""
        from src.models.performance_logger import PerformanceLogger
        import json
        from datetime import datetime

        logger = PerformanceLogger()
        logger.LOG_DIR = tmp_path

        # Write some test data
        today = datetime.utcnow().strftime("%Y%m%d")
        filepath = tmp_path / f"predictions_{today}.jsonl"

        entries = [
            {"match_id": "1", "market_type": "ou", "poisson_prob": 0.6, "ml_prob": 0.65, "true_edge": 0.08, "is_actionable": True},
            {"match_id": "2", "market_type": "btts", "poisson_prob": 0.5, "ml_prob": 0.55, "true_edge": 0.03, "is_actionable": False},
        ]
        with open(filepath, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        stats = logger.get_daily_stats()

        assert stats["n_predictions"] == 2
        assert stats["n_actionable"] == 1


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_edge_detector_basic_flow(self):
        """Basic EdgeDetector should work without ML."""
        detector = EdgeDetector()

        # Create mock probabilities
        fair_probs = MagicMock()
        fair_probs.fair_1x2 = {"home": 0.45, "draw": 0.28, "away": 0.27}
        fair_probs.fair_ou = {"2.5": {"over": 0.55, "under": 0.45}}
        fair_probs.fair_btts = {"yes": 0.52, "no": 0.48}

        market_probs = MagicMock()
        market_probs.polymarket_1x2 = {"home": 0.40, "draw": 0.30, "away": 0.30}
        market_probs.polymarket_ou = {"2.5": {"over": 0.50, "under": 0.50}}
        market_probs.polymarket_btts = {"yes": 0.50, "no": 0.50}

        analysis = detector.analyze_match(
            match_id="test",
            home_team="Arsenal",
            away_team="Chelsea",
            fair_probs=fair_probs,
            market_probs=market_probs,
        )

        assert isinstance(analysis, MatchEdgeAnalysis)
        assert analysis.match_id == "test"
        assert analysis.all_edges is not None

    def test_hybrid_detector_falls_back_to_poisson(self):
        """HybridEdgeDetector should fall back to Poisson when ML unavailable."""
        detector = HybridEdgeDetector(use_ml=False)

        fair_probs = MagicMock()
        fair_probs.fair_1x2 = {"home": 0.45, "draw": 0.28, "away": 0.27}
        fair_probs.fair_ou = {"2.5": {"over": 0.55, "under": 0.45}}
        fair_probs.fair_btts = {"yes": 0.52, "no": 0.48}

        market_probs = MagicMock()
        market_probs.polymarket_1x2 = {"home": 0.40, "draw": 0.30, "away": 0.30}
        market_probs.polymarket_ou = {"2.5": {"over": 0.50, "under": 0.50}}
        market_probs.polymarket_btts = {"yes": 0.50, "no": 0.50}

        # Should not raise, should return base analysis
        analysis = detector.analyze_match_hybrid(
            match_id="test",
            home_team="Arsenal",
            away_team="Chelsea",
            fair_probs=fair_probs,
            market_probs=market_probs,
            features=None,
        )

        assert isinstance(analysis, MatchEdgeAnalysis)


class TestClassificationResultIntegration:
    """Test ClassificationResult integration with HybridEdgeDetector."""

    def test_classification_result_used_by_detector(self):
        """ClassificationResult should integrate with _enhance_with_ml."""
        detector = HybridEdgeDetector(use_ml=False)

        base_signal = EdgeSignal(
            market_type="ou",
            outcome="over",
            fair_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            edge_pct=8.0,
            direction=SignalDirection.OVER,
            is_actionable=True,
            threshold_used=0.07,
        )

        ml_result = ClassificationResult(
            market_type="ou",
            outcome="over",
            probability=0.65,
            confidence="high",
            model_used="ensemble",
        )

        enhanced = detector._enhance_with_ml(
            base_signal=base_signal,
            ml_result=ml_result,
            market_type="ou",
            home_team="Arsenal",
            away_team="Chelsea",
        )

        assert isinstance(enhanced, HybridEdgeSignal)
        assert enhanced.model_source == ModelSource.HYBRID
        assert enhanced.ml_confidence == "high"
        assert enhanced.poisson_prob == 0.58
        assert enhanced.ml_prob == 0.65
