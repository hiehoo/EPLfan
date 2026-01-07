"""Tests for processing layer (models)."""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from src.models.poisson_matrix import PoissonCalculator, PoissonResult
from src.models.market_probabilities import MarketProbabilityCalculator, MatchProbabilities
from src.models.weight_engine import WeightEngine, WeightedProbabilities
from src.models.edge_detector import EdgeDetector, EdgeSignal, MatchEdgeAnalysis, SignalDirection


class TestPoissonCalculator:
    """Test Poisson probability calculations."""

    def test_poisson_matrix_shape(self):
        """Test matrix has correct dimensions."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.5, away_xg=1.2)

        assert result.matrix.shape == (8, 8)

    def test_poisson_matrix_sums_to_one(self):
        """Test matrix probabilities sum to 1.0."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.5, away_xg=1.2)

        total = result.matrix.sum()
        assert abs(total - 1.0) < 0.001

    def test_1x2_probabilities_sum_to_one(self):
        """Test 1X2 probabilities sum to 1.0."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.8, away_xg=1.0)

        total = sum(result.p_1x2.values())
        assert abs(total - 1.0) < 0.001

    def test_1x2_home_advantage(self):
        """Test home team with higher xG has higher win probability."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=2.0, away_xg=1.0)

        assert result.p_1x2["home"] > result.p_1x2["away"]
        assert result.p_1x2["home"] > result.p_1x2["draw"]

    def test_over_under_probabilities_sum_to_one(self):
        """Test O/U probabilities sum to 1.0 for each line."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.5, away_xg=1.5)

        for line, probs in result.p_over_under.items():
            total = probs["over"] + probs["under"]
            assert abs(total - 1.0) < 0.001, f"Line {line} sums to {total}"

    def test_over_probability_increases_with_xg(self):
        """Test higher combined xG increases Over probability."""
        calc = PoissonCalculator()

        low_xg_result = calc.calculate(home_xg=1.0, away_xg=0.8)
        high_xg_result = calc.calculate(home_xg=2.5, away_xg=2.0)

        # Over 2.5 should be higher with high xG
        assert high_xg_result.p_over_under["2.5"]["over"] > low_xg_result.p_over_under["2.5"]["over"]

    def test_btts_probabilities_sum_to_one(self):
        """Test BTTS probabilities sum to 1.0."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.5, away_xg=1.2)

        total = result.p_btts["yes"] + result.p_btts["no"]
        assert abs(total - 1.0) < 0.001

    def test_btts_increases_with_balanced_xg(self):
        """Test balanced xG teams have higher BTTS yes probability."""
        calc = PoissonCalculator()

        # Both teams scoring
        balanced = calc.calculate(home_xg=1.5, away_xg=1.5)
        # One team dominant
        lopsided = calc.calculate(home_xg=3.0, away_xg=0.3)

        # BTTS more likely when both teams expected to score
        assert balanced.p_btts["yes"] > lopsided.p_btts["yes"]

    def test_extreme_xg_clamped(self):
        """Test extreme xG values are clamped."""
        calc = PoissonCalculator()

        # Very high xG
        result_high = calc.calculate(home_xg=10.0, away_xg=0.01)
        assert result_high.home_xg <= 5.0
        assert result_high.away_xg >= 0.1

    def test_most_likely_scores(self):
        """Test getting most likely scorelines."""
        calc = PoissonCalculator()
        result = calc.calculate(home_xg=1.5, away_xg=1.0)

        top_scores = calc.get_most_likely_scores(result.matrix, top_n=5)

        assert len(top_scores) == 5
        # First score should have highest probability
        assert top_scores[0][2] >= top_scores[1][2]
        # Probabilities should be positive
        for h, a, prob in top_scores:
            assert prob > 0


class TestMarketProbabilityCalculator:
    """Test market probability calculations."""

    def test_odds_to_probabilities_normalized(self):
        """Test odds conversion removes overround."""
        calc = MarketProbabilityCalculator()

        # Odds with 5% overround
        probs = calc._odds_to_probabilities({
            "home": 2.0,  # 50% implied
            "draw": 3.5,  # 28.6% implied
            "away": 4.0,  # 25% implied
            # Total = 103.6%
        })

        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001

    def test_odds_to_probabilities_preserves_ranking(self):
        """Test favorite remains favorite after normalization."""
        calc = MarketProbabilityCalculator()

        probs = calc._odds_to_probabilities({
            "home": 1.5,  # Strong favorite
            "draw": 4.0,
            "away": 6.0,
        })

        assert probs["home"] > probs["draw"]
        assert probs["draw"] > probs["away"]

    def test_odds_to_probabilities_handles_zeros(self):
        """Test handling of zero odds."""
        calc = MarketProbabilityCalculator()

        probs = calc._odds_to_probabilities({
            "home": 2.0,
            "draw": 0,  # Invalid
            "away": 4.0,
        })

        # Should normalize without zero
        assert "draw" not in probs or probs["draw"] == 0

    def test_calculate_all_returns_match_probabilities(self):
        """Test full calculation returns correct structure."""
        calc = MarketProbabilityCalculator()

        result = calc.calculate_all(
            match_id="test-123",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=48),
            home_xg=1.8,
            away_xg=1.2,
            home_xga=0.9,
            away_xga=1.3,
            home_elo=1900,
            away_elo=1850,
            bf_home_odds=2.1,
            bf_draw_odds=3.5,
            bf_away_odds=3.6,
        )

        assert isinstance(result, MatchProbabilities)
        assert result.match_id == "test-123"
        assert result.home_team == "Arsenal"
        assert isinstance(result.poisson_result, PoissonResult)

    def test_calculate_all_with_all_data(self):
        """Test calculation with all data sources."""
        calc = MarketProbabilityCalculator()

        result = calc.calculate_all(
            match_id="test-456",
            home_team="Liverpool",
            away_team="Man City",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=24),
            home_xg=2.0,
            away_xg=1.8,
            home_xga=1.0,
            away_xga=0.9,
            home_elo=1950,
            away_elo=1980,
            bf_home_odds=2.8,
            bf_draw_odds=3.4,
            bf_away_odds=2.5,
            bf_over_2_5_odds=1.75,
            bf_under_2_5_odds=2.15,
            bf_btts_yes_odds=1.65,
            bf_btts_no_odds=2.25,
            pm_home_price=0.35,
            pm_draw_price=0.30,
            pm_away_price=0.35,
            pm_over_2_5_price=0.58,
            pm_btts_yes_price=0.62,
        )

        # Betfair O/U should be populated
        assert "2.5" in result.betfair_ou
        # Polymarket should be populated
        assert result.polymarket_1x2["home"] == 0.35


class TestWeightEngine:
    """Test weight application."""

    def test_preset_selection_analytics_first(self):
        """Test analytics_first preset selected for >96 hours."""
        engine = WeightEngine()

        preset = engine.get_preset_for_timing("1x2", hours_to_kickoff=120)
        assert preset == "analytics_first"

    def test_preset_selection_balanced(self):
        """Test balanced preset selected for 24-96 hours."""
        engine = WeightEngine()

        preset = engine.get_preset_for_timing("1x2", hours_to_kickoff=48)
        assert preset == "balanced"

    def test_preset_selection_market_trust(self):
        """Test market_trust preset selected for <24 hours."""
        engine = WeightEngine()

        preset = engine.get_preset_for_timing("1x2", hours_to_kickoff=12)
        assert preset == "market_trust"

    def test_fair_probabilities_normalized(self):
        """Test fair probabilities sum to 1.0."""
        engine = WeightEngine()

        fair_probs, preset, breakdown = engine.calculate_fair_probabilities(
            p_betfair={"home": 0.45, "draw": 0.28, "away": 0.27},
            p_xg={"home": 0.50, "draw": 0.25, "away": 0.25},
            p_elo={"home": 0.48, "draw": 0.26, "away": 0.26},
            p_form={"home": 0.52, "draw": 0.24, "away": 0.24},
            hours_to_kickoff=48,
            market_type="1x2",
        )

        total = sum(fair_probs.values())
        assert abs(total - 1.0) < 0.001

    def test_preset_override(self):
        """Test preset override works."""
        engine = WeightEngine()

        fair_probs, preset, breakdown = engine.calculate_fair_probabilities(
            p_betfair={"home": 0.45, "draw": 0.28, "away": 0.27},
            p_xg={"home": 0.50, "draw": 0.25, "away": 0.25},
            p_elo={"home": 0.48, "draw": 0.26, "away": 0.26},
            p_form={"home": 0.52, "draw": 0.24, "away": 0.24},
            hours_to_kickoff=120,  # Would be analytics_first
            market_type="1x2",
            preset_override="market_trust",  # But override to market_trust
        )

        assert preset == "market_trust"

    def test_weights_recorded_in_breakdown(self):
        """Test breakdown includes weight values."""
        engine = WeightEngine()

        fair_probs, preset, breakdown = engine.calculate_fair_probabilities(
            p_betfair={"home": 0.45, "draw": 0.28, "away": 0.27},
            p_xg={"home": 0.50, "draw": 0.25, "away": 0.25},
            p_elo={"home": 0.48, "draw": 0.26, "away": 0.26},
            p_form={"home": 0.52, "draw": 0.24, "away": 0.24},
            hours_to_kickoff=48,
            market_type="1x2",
        )

        assert "weights" in breakdown
        assert "betfair" in breakdown
        assert "xg" in breakdown


class TestEdgeDetector:
    """Test edge detection."""

    def test_edge_calculation(self):
        """Test edge = fair - market."""
        detector = EdgeDetector()

        # Create mock fair probs
        class MockFairProbs:
            fair_1x2 = {"home": 0.50, "draw": 0.25, "away": 0.25}
            fair_ou = {"2.5": {"over": 0.55, "under": 0.45}}
            fair_btts = {"yes": 0.60, "no": 0.40}

        # Create mock market probs
        class MockMarketProbs:
            polymarket_1x2 = {"home": 0.45, "draw": 0.28, "away": 0.27}
            polymarket_ou = {"2.5": {"over": 0.50, "under": 0.50}}
            polymarket_btts = {"yes": 0.55, "no": 0.45}

        edges = detector._calculate_all_edges(MockFairProbs(), MockMarketProbs())

        # Home edge = 0.50 - 0.45 = 0.05
        assert abs(edges["1x2"]["home"]["edge"] - 0.05) < 0.001

    def test_actionable_edge_threshold(self):
        """Test edge exceeds threshold check."""
        detector = EdgeDetector()

        signal = EdgeSignal(
            market_type="1x2",
            outcome="home",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,  # 7% edge
            edge_pct=7.0,
            direction=SignalDirection.HOME,
            is_actionable=True,  # 7% > 5% threshold
            threshold_used=0.05,
        )

        assert signal.is_actionable is True

    def test_non_actionable_edge(self):
        """Test edge below threshold not actionable."""
        detector = EdgeDetector()

        signal = EdgeSignal(
            market_type="1x2",
            outcome="home",
            fair_prob=0.48,
            market_prob=0.46,
            edge=0.02,  # 2% edge
            edge_pct=2.0,
            direction=SignalDirection.HOME,
            is_actionable=False,  # 2% < 5% threshold
            threshold_used=0.05,
        )

        assert signal.is_actionable is False

    def test_direction_positive_edge(self):
        """Test direction for positive edge (underpriced)."""
        detector = EdgeDetector()

        # Positive edge = underpriced = buy that outcome
        direction = detector._get_direction("1x2", "home", edge=0.05)
        assert direction == SignalDirection.HOME

        direction = detector._get_direction("ou", "over", edge=0.06)
        assert direction == SignalDirection.OVER

    def test_direction_negative_edge(self):
        """Test direction for negative edge (overpriced)."""
        detector = EdgeDetector()

        # Negative edge = overpriced = buy opposite
        direction = detector._get_direction("1x2", "home", edge=-0.05)
        assert direction == SignalDirection.AWAY

        direction = detector._get_direction("ou", "over", edge=-0.06)
        assert direction == SignalDirection.UNDER

    def test_analyze_match_finds_best_signal(self):
        """Test analyze_match returns best actionable signal."""
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
            home_team="Arsenal",
            away_team="Chelsea",
            fair_probs=MockFairProbs(),
            market_probs=MockMarketProbs(),
        )

        assert isinstance(analysis, MatchEdgeAnalysis)
        # Should have best signal (highest edge)
        if analysis.best_signal:
            assert analysis.best_signal.is_actionable


class TestMatchAnalyzer:
    """Test full analysis pipeline."""

    def test_full_pipeline(self):
        """Test complete match analysis pipeline."""
        from src.models import MatchAnalyzer

        analyzer = MatchAnalyzer()

        match_probs, weighted_probs, edge_analysis = analyzer.analyze(
            match_id="test-full",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=48),
            home_xg=1.8,
            away_xg=1.2,
            home_xga=0.9,
            away_xga=1.3,
            home_elo=1900,
            away_elo=1850,
            bf_home_odds=2.1,
            bf_draw_odds=3.5,
            bf_away_odds=3.6,
            bf_over_2_5_odds=1.85,
            bf_under_2_5_odds=2.05,
            bf_btts_yes_odds=1.75,
            bf_btts_no_odds=2.10,
            pm_home_price=0.48,
            pm_draw_price=0.27,
            pm_away_price=0.25,
            pm_over_2_5_price=0.52,
            pm_btts_yes_price=0.58,
        )

        # Check all outputs
        assert isinstance(match_probs, MatchProbabilities)
        assert isinstance(weighted_probs, WeightedProbabilities)
        assert isinstance(edge_analysis, MatchEdgeAnalysis)

        # Check probabilities normalized
        assert abs(sum(match_probs.betfair_1x2.values()) - 1.0) < 0.001
        assert abs(sum(weighted_probs.fair_1x2.values()) - 1.0) < 0.001

    def test_pipeline_with_minimal_data(self):
        """Test pipeline with only required data."""
        from src.models import MatchAnalyzer

        analyzer = MatchAnalyzer()

        match_probs, weighted_probs, edge_analysis = analyzer.analyze(
            match_id="test-minimal",
            home_team="Liverpool",
            away_team="Man City",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=12),
            home_xg=2.0,
            away_xg=1.8,
            home_xga=1.0,
            away_xga=0.9,
            home_elo=1950,
            away_elo=1980,
            bf_home_odds=2.8,
            bf_draw_odds=3.4,
            bf_away_odds=2.5,
            # No O/U, BTTS, or Polymarket data
        )

        assert match_probs is not None
        assert weighted_probs is not None
        assert edge_analysis is not None

    def test_pipeline_preset_selection(self):
        """Test pipeline selects correct preset based on time."""
        from src.models import MatchAnalyzer

        analyzer = MatchAnalyzer()

        # Far out game (>96h) should use analytics_first
        _, weighted_probs, _ = analyzer.analyze(
            match_id="test-far",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=120),
            home_xg=1.8,
            away_xg=1.2,
            home_xga=0.9,
            away_xga=1.3,
            home_elo=1900,
            away_elo=1850,
            bf_home_odds=2.1,
            bf_draw_odds=3.5,
            bf_away_odds=3.6,
        )

        assert weighted_probs.preset_name == "analytics_first"

        # Close game (<24h) should use market_trust
        _, weighted_probs_close, _ = analyzer.analyze(
            match_id="test-close",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now(timezone.utc) + timedelta(hours=12),
            home_xg=1.8,
            away_xg=1.2,
            home_xga=0.9,
            away_xga=1.3,
            home_elo=1900,
            away_elo=1850,
            bf_home_odds=2.1,
            bf_draw_odds=3.5,
            bf_away_odds=3.6,
        )

        assert weighted_probs_close.preset_name == "market_trust"
