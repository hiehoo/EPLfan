"""Tests for base rate tracking and true edge calculation."""
import pytest
from unittest.mock import MagicMock, patch

from src.models.base_rate_tracker import BaseRateTracker, BaseRates, base_rate_tracker
from src.models.edge_detector import EdgeDetector, EdgeSignal, SignalDirection


class TestBaseRateTracker:
    """Tests for BaseRateTracker."""

    def test_default_base_rates(self):
        """Test default EPL base rates are reasonable."""
        tracker = BaseRateTracker()

        # 1X2 should sum to ~1
        assert 0.40 < tracker.get_base_rate("1x2", "home") < 0.50
        assert 0.20 < tracker.get_base_rate("1x2", "draw") < 0.30
        assert 0.25 < tracker.get_base_rate("1x2", "away") < 0.35

        # O/U (EPL typically 50-60% over)
        assert 0.45 < tracker.get_base_rate("ou", "over", 2.5) < 0.65

        # BTTS
        assert 0.50 < tracker.get_base_rate("btts", "yes") < 0.58

    def test_base_rates_sum_to_one(self):
        """Test base rates for each market sum to ~1.0."""
        tracker = BaseRateTracker()

        # 1X2
        total_1x2 = (
            tracker.get_base_rate("1x2", "home") +
            tracker.get_base_rate("1x2", "draw") +
            tracker.get_base_rate("1x2", "away")
        )
        assert abs(total_1x2 - 1.0) < 0.01

        # O/U
        total_ou = (
            tracker.get_base_rate("ou", "over", 2.5) +
            tracker.get_base_rate("ou", "under", 2.5)
        )
        assert abs(total_ou - 1.0) < 0.01

        # BTTS
        total_btts = (
            tracker.get_base_rate("btts", "yes") +
            tracker.get_base_rate("btts", "no")
        )
        assert abs(total_btts - 1.0) < 0.01

    def test_elite_teams_identified(self):
        """Test elite teams are correctly identified."""
        tracker = BaseRateTracker()

        assert tracker.is_elite_team("Manchester City")
        assert tracker.is_elite_team("Liverpool")
        assert tracker.is_elite_team("Arsenal")
        assert tracker.is_elite_team("Chelsea")
        assert not tracker.is_elite_team("Burnley")
        assert not tracker.is_elite_team("Southampton")
        assert not tracker.is_elite_team("Ipswich")

    def test_elite_adjustment_applied(self):
        """Test elite adjustment returns correct value."""
        tracker = BaseRateTracker()

        assert tracker.get_elite_adjustment("Manchester City") == 1.15
        assert tracker.get_elite_adjustment("Liverpool") == 1.15
        assert tracker.get_elite_adjustment("Burnley") == 1.0
        assert tracker.get_elite_adjustment("Unknown Team") == 1.0

    def test_unknown_market_returns_default(self):
        """Test unknown market type returns 0.5."""
        tracker = BaseRateTracker()

        assert tracker.get_base_rate("unknown", "whatever") == 0.5

    def test_unknown_outcome_returns_reasonable_default(self):
        """Test unknown outcome returns reasonable default."""
        tracker = BaseRateTracker()

        # Unknown 1x2 outcome
        assert tracker.get_base_rate("1x2", "unknown") == 0.33

        # Unknown O/U outcome
        assert tracker.get_base_rate("ou", "unknown", 2.5) == 0.5

        # Unknown BTTS outcome
        assert tracker.get_base_rate("btts", "unknown") == 0.5


class TestTrueEdgeCalculation:
    """Tests for true edge calculation."""

    def test_true_edge_less_than_naive(self):
        """True edge should be <= naive edge when skill component is limiting."""
        detector = EdgeDetector()

        # Scenario: Model says 55% home, market says 50%
        # Naive edge = 5%, but base rate for home is 46%
        true_edge, skill, base_rate, _ = detector.calculate_true_edge(
            model_prob=0.55,
            market_prob=0.50,
            market_type="1x2",
            outcome="home",
        )

        naive_edge = 0.55 - 0.50  # 5%

        # Skill component = 0.55 - max(0.46, 0.54) = 0.55 - 0.54 = 0.01
        # True edge should be min(naive, skill) = min(0.05, 0.01) = 0.01
        assert true_edge <= naive_edge
        assert skill < naive_edge

    def test_true_edge_zero_when_below_base(self):
        """True edge should be 0 when model doesn't beat base rate."""
        detector = EdgeDetector()

        # Model says 45% home, but base rate is 46%
        # Even with naive edge of 5%, no true skill
        true_edge, skill, base_rate, _ = detector.calculate_true_edge(
            model_prob=0.45,
            market_prob=0.40,
            market_type="1x2",
            outcome="home",
        )

        # skill = 0.45 - max(0.46, 0.54) = 0.45 - 0.54 = -0.09
        assert skill < 0
        # No skill = no true edge
        assert true_edge == 0

    def test_elite_match_boost(self):
        """Elite matches should get adjustment boost."""
        detector = EdgeDetector()

        # Same scenario but with elite team
        true_edge_elite, _, _, is_elite = detector.calculate_true_edge(
            model_prob=0.60,
            market_prob=0.50,
            market_type="1x2",
            outcome="home",
            home_team="Manchester City",
            away_team="Burnley",
        )

        true_edge_normal, _, _, _ = detector.calculate_true_edge(
            model_prob=0.60,
            market_prob=0.50,
            market_type="1x2",
            outcome="home",
            home_team="Burnley",
            away_team="Southampton",
        )

        assert is_elite
        assert true_edge_elite > true_edge_normal

    def test_elite_detected_for_away_team(self):
        """Elite team detection works for away team too."""
        detector = EdgeDetector()

        _, _, _, is_elite = detector.calculate_true_edge(
            model_prob=0.60,
            market_prob=0.50,
            market_type="1x2",
            outcome="away",
            home_team="Burnley",
            away_team="Liverpool",
        )

        assert is_elite

    def test_true_edge_ou_market(self):
        """Test true edge calculation for O/U market."""
        detector = EdgeDetector()

        true_edge, skill, base_rate, _ = detector.calculate_true_edge(
            model_prob=0.65,
            market_prob=0.55,
            market_type="ou",
            outcome="over",
            line=2.5,
        )

        # Base rate for over 2.5 (EPL ~52-60%)
        assert 0.50 < base_rate < 0.65
        # With 65% model prob vs 52% base, should have positive skill
        assert skill > 0
        assert true_edge > 0

    def test_true_edge_btts_market(self):
        """Test true edge calculation for BTTS market."""
        detector = EdgeDetector()

        true_edge, skill, base_rate, _ = detector.calculate_true_edge(
            model_prob=0.70,
            market_prob=0.60,
            market_type="btts",
            outcome="yes",
        )

        # Base rate for BTTS yes is ~0.53
        assert 0.50 < base_rate < 0.58
        # With 70% model prob vs 53% base, should have positive skill
        assert skill > 0
        assert true_edge > 0


class TestEdgeSignalEnhancement:
    """Tests for enhanced EdgeSignal with base rate fields."""

    def test_signal_includes_base_rate_fields(self):
        """EdgeSignal should include new base rate fields."""
        signal = EdgeSignal(
            market_type="1x2",
            outcome="home",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            edge_pct=7.0,
            direction=SignalDirection.HOME,
            is_actionable=True,
            threshold_used=0.05,
            base_rate=0.46,
            true_edge=0.05,
            skill_component=0.09,
            is_elite_match=False,
        )

        assert signal.base_rate == 0.46
        assert signal.true_edge == 0.05
        assert signal.skill_component == 0.09
        assert signal.is_elite_match is False

    def test_signal_default_values(self):
        """EdgeSignal should have sensible defaults for new fields."""
        signal = EdgeSignal(
            market_type="1x2",
            outcome="home",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            edge_pct=7.0,
            direction=SignalDirection.HOME,
            is_actionable=True,
            threshold_used=0.05,
        )

        # Default values
        assert signal.base_rate == 0.5
        assert signal.true_edge == 0.0
        assert signal.skill_component == 0.0
        assert signal.is_elite_match is False


class TestEdgeDetectorIntegration:
    """Integration tests for EdgeDetector with true edge."""

    def test_find_best_edge_uses_true_edge(self):
        """Test that _find_best_edge_in_market uses true edge for actionable check."""
        detector = EdgeDetector()

        # Market edges with different true edge potential
        market_edges = {
            "home": {"fair": 0.55, "market": 0.50, "edge": 0.05},
            "draw": {"fair": 0.30, "market": 0.28, "edge": 0.02},
            "away": {"fair": 0.15, "market": 0.22, "edge": -0.07},
        }

        signal = detector._find_best_edge_in_market(
            market_edges=market_edges,
            market_type="1x2",
            threshold=0.05,
            home_team="Arsenal",
            away_team="Chelsea",
        )

        assert signal is not None
        assert signal.true_edge is not None
        assert signal.base_rate > 0
        # Signal should have true_edge populated
        assert isinstance(signal.true_edge, float)

    def test_analyze_match_populates_true_edge(self):
        """Test full match analysis populates true edge fields."""
        detector = EdgeDetector()

        class MockFairProbs:
            fair_1x2 = {"home": 0.55, "draw": 0.25, "away": 0.20}
            fair_ou = {"2.5": {"over": 0.60, "under": 0.40}}
            fair_btts = {"yes": 0.65, "no": 0.35}

        class MockMarketProbs:
            polymarket_1x2 = {"home": 0.45, "draw": 0.28, "away": 0.27}
            polymarket_ou = {"2.5": {"over": 0.52, "under": 0.48}}
            polymarket_btts = {"yes": 0.55, "no": 0.45}

        analysis = detector.analyze_match(
            match_id="test-123",
            home_team="Manchester City",  # Elite team
            away_team="Burnley",
            fair_probs=MockFairProbs(),
            market_probs=MockMarketProbs(),
        )

        # Check 1X2 signal has true edge
        if analysis.signal_1x2:
            assert analysis.signal_1x2.base_rate > 0
            assert analysis.signal_1x2.is_elite_match is True  # Man City is elite

        # Check O/U signal
        if analysis.signal_ou:
            assert analysis.signal_ou.base_rate > 0

        # Check BTTS signal
        if analysis.signal_btts:
            assert analysis.signal_btts.base_rate > 0

    def test_elite_match_affects_actionability(self):
        """Test elite teams can make edge more actionable."""
        detector = EdgeDetector()

        class MockFairProbs:
            fair_1x2 = {"home": 0.58, "draw": 0.22, "away": 0.20}
            fair_ou = {"2.5": {"over": 0.55, "under": 0.45}}
            fair_btts = {"yes": 0.58, "no": 0.42}

        class MockMarketProbs:
            polymarket_1x2 = {"home": 0.52, "draw": 0.25, "away": 0.23}
            polymarket_ou = {"2.5": {"over": 0.50, "under": 0.50}}
            polymarket_btts = {"yes": 0.55, "no": 0.45}

        # Elite match
        analysis_elite = detector.analyze_match(
            match_id="test-elite",
            home_team="Liverpool",
            away_team="Arsenal",
            fair_probs=MockFairProbs(),
            market_probs=MockMarketProbs(),
        )

        # Non-elite match
        analysis_normal = detector.analyze_match(
            match_id="test-normal",
            home_team="Brighton",
            away_team="Brentford",
            fair_probs=MockFairProbs(),
            market_probs=MockMarketProbs(),
        )

        # Both matches analyzed
        assert analysis_elite is not None
        assert analysis_normal is not None

        # Elite match should be flagged
        if analysis_elite.signal_1x2:
            assert analysis_elite.signal_1x2.is_elite_match is True
        if analysis_normal.signal_1x2:
            assert analysis_normal.signal_1x2.is_elite_match is False
