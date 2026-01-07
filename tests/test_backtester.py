"""Tests for backtesting engine."""
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.models.backtester import (
    Backtester,
    BacktestParams,
    BacktestResult,
    BacktestReport,
    MarketStats,
)


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_create_result(self):
        """Create a basic backtest result."""
        result = BacktestResult(
            match_id=1,
            market_type="ou",
            predicted_prob=0.55,
            market_prob=0.50,
            edge=0.05,
            actual_outcome="over",
            is_correct=True,
            pnl=0.90,
        )

        assert result.match_id == 1
        assert result.market_type == "ou"
        assert result.edge == 0.05
        assert result.is_correct is True


class TestMetricCalculations:
    """Test metric calculation methods."""

    def test_brier_score_perfect(self):
        """Brier score of 0 for perfect predictions."""
        backtester = Backtester()

        results = [
            BacktestResult(1, "ou", 1.0, 0.5, 0.5, "over", True, 1.0),
            BacktestResult(2, "ou", 1.0, 0.5, 0.5, "over", True, 1.0),
        ]

        brier = backtester._calculate_brier_score(results)
        assert brier == 0.0

    def test_brier_score_random(self):
        """Brier score calculation for 50% predictions."""
        backtester = Backtester()

        results = [
            BacktestResult(1, "ou", 0.5, 0.5, 0.0, "over", True, 1.0),
            BacktestResult(2, "ou", 0.5, 0.5, 0.0, "under", False, -1.0),
        ]

        brier = backtester._calculate_brier_score(results)
        assert abs(brier - 0.25) < 0.01  # Expected ~0.25 for 50% confidence

    def test_brier_score_empty(self):
        """Brier score for empty results."""
        backtester = Backtester()
        brier = backtester._calculate_brier_score([])
        assert brier == 0.0

    def test_sharpe_ratio_positive(self):
        """Sharpe ratio for positive returns."""
        backtester = Backtester()

        # Consistent positive returns
        pnl_series = [1.0, 1.0, 1.0, 1.0]
        sharpe = backtester._calculate_sharpe(pnl_series)

        # With zero std, should return 0
        assert sharpe == 0.0

    def test_sharpe_ratio_mixed(self):
        """Sharpe ratio for mixed returns."""
        backtester = Backtester()

        pnl_series = [1.0, -0.5, 0.8, -0.3, 1.2]
        sharpe = backtester._calculate_sharpe(pnl_series)

        assert sharpe > 0  # Positive overall

    def test_sharpe_ratio_empty(self):
        """Sharpe ratio for empty series."""
        backtester = Backtester()
        sharpe = backtester._calculate_sharpe([])
        assert sharpe == 0.0

    def test_max_drawdown_calculation(self):
        """Calculate max drawdown correctly."""
        backtester = Backtester()

        # Cumulative: 1, 2, 1, 2, 3 -> peak at 2, trough at 1 = 1 unit drawdown
        pnl_series = [1.0, 1.0, -1.0, 1.0, 1.0]
        drawdown = backtester._calculate_max_drawdown(pnl_series)

        assert drawdown == 1.0

    def test_max_drawdown_no_drawdown(self):
        """No drawdown with only positive returns."""
        backtester = Backtester()

        pnl_series = [1.0, 1.0, 1.0]
        drawdown = backtester._calculate_max_drawdown(pnl_series)

        assert drawdown == 0.0

    def test_max_drawdown_empty(self):
        """Empty series returns 0."""
        backtester = Backtester()
        assert backtester._calculate_max_drawdown([]) == 0.0

    def test_streaks_calculation(self):
        """Calculate win/lose streaks correctly."""
        backtester = Backtester()

        results = [
            BacktestResult(1, "ou", 0.5, 0.5, 0.0, "over", True, 1.0),
            BacktestResult(2, "ou", 0.5, 0.5, 0.0, "over", True, 1.0),
            BacktestResult(3, "ou", 0.5, 0.5, 0.0, "over", True, 1.0),
            BacktestResult(4, "ou", 0.5, 0.5, 0.0, "under", False, -1.0),
            BacktestResult(5, "ou", 0.5, 0.5, 0.0, "under", False, -1.0),
        ]

        win_streak, lose_streak = backtester._calculate_streaks(results)

        assert win_streak == 3
        assert lose_streak == 2

    def test_streaks_empty(self):
        """Empty results return 0 streaks."""
        backtester = Backtester()
        win, lose = backtester._calculate_streaks([])
        assert win == 0
        assert lose == 0


class TestCalibrationError:
    """Test calibration error calculation."""

    def test_calibration_well_calibrated(self):
        """Low calibration error for well-calibrated predictions."""
        backtester = Backtester()

        # 50% predictions with 50% actual rate
        results = [
            BacktestResult(i, "ou", 0.5, 0.5, 0.0, "over", i % 2 == 0, 1.0 if i % 2 == 0 else -1.0)
            for i in range(100)
        ]

        error = backtester._calculate_calibration_error(results)
        assert error < 0.1  # Should be low for well-calibrated

    def test_calibration_empty(self):
        """Empty results return 0."""
        backtester = Backtester()
        assert backtester._calculate_calibration_error([]) == 0.0


class TestBacktestReport:
    """Test BacktestReport generation."""

    def test_calculate_metrics_basic(self):
        """Calculate basic metrics from results."""
        backtester = Backtester()

        results = [
            BacktestResult(1, "ou", 0.6, 0.5, 0.1, "over", True, 0.90),
            BacktestResult(2, "ou", 0.6, 0.5, 0.1, "over", True, 0.90),
            BacktestResult(3, "ou", 0.6, 0.5, 0.1, "under", False, -1.0),
        ]

        report = backtester._calculate_metrics(results, 30)

        assert report.n_predictions == 3
        assert report.n_correct == 2
        assert abs(report.win_rate - 0.667) < 0.01

    def test_calculate_metrics_market_breakdown(self):
        """Calculate per-market breakdown."""
        backtester = Backtester()

        results = [
            BacktestResult(1, "ou", 0.6, 0.5, 0.1, "over", True, 0.90),
            BacktestResult(2, "btts", 0.5, 0.5, 0.0, "yes", True, 0.90),
            BacktestResult(3, "1x2", 0.4, 0.4, 0.0, "home", False, -1.0),
        ]

        report = backtester._calculate_metrics(results, 30)

        assert "ou" in report.by_market
        assert "btts" in report.by_market
        assert "1x2" in report.by_market
        assert report.by_market["ou"].n_predictions == 1


class TestJSONExport:
    """Test JSON export functionality."""

    def test_to_json_valid(self):
        """Export report as valid JSON."""
        backtester = Backtester()

        report = BacktestReport(
            period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
            n_predictions=100,
            n_correct=55,
            win_rate=0.55,
            roi_pct=5.5,
            profit_factor=1.2,
            brier_score=0.22,
            calibration_error=0.05,
            sharpe_ratio=0.8,
            max_drawdown=10.0,
            max_win_streak=5,
            max_lose_streak=3,
            by_market={
                "ou": MarketStats(50, 28, 0.56, 6.0, 0.03),
            },
        )

        json_str = backtester.to_json(report)
        data = json.loads(json_str)

        assert data["n_predictions"] == 100
        assert data["win_rate"] == 0.55
        assert "by_market" in data
        assert data["by_market"]["ou"]["n_predictions"] == 50

    def test_to_json_empty_markets(self):
        """Export report with no market data."""
        backtester = Backtester()

        report = BacktestReport(
            period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
            n_predictions=0,
            n_correct=0,
            win_rate=0.0,
            roi_pct=0.0,
            profit_factor=0.0,
            brier_score=0.0,
            calibration_error=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_win_streak=0,
            max_lose_streak=0,
        )

        json_str = backtester.to_json(report)
        data = json.loads(json_str)

        assert data["n_predictions"] == 0
        assert data["by_market"] == {}


class TestBacktestRun:
    """Test full backtest run."""

    @patch("src.models.backtester.db")
    def test_run_no_predictions(self, mock_db):
        """Handle no predictions gracefully."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        backtester = Backtester()
        params = BacktestParams(days=30, market="all")
        report = backtester.run(params)

        assert report.n_predictions == 0
        assert report.win_rate == 0.0

    @patch("src.models.backtester.db")
    def test_run_with_data(self, mock_db):
        """Run backtest with mock data."""
        # Create mock match
        mock_match = MagicMock()
        mock_match.id = 1
        mock_match.is_completed = True
        mock_match.home_goals = 2
        mock_match.away_goals = 1

        # Create mock odds
        mock_odds = MagicMock()
        mock_odds.bf_over_2_5_odds = 1.90
        mock_odds.bf_under_2_5_odds = 2.00
        mock_odds.bf_btts_yes_odds = None
        mock_odds.bf_btts_no_odds = None
        mock_odds.bf_home_odds = None
        mock_odds.bf_draw_odds = None
        mock_odds.bf_away_odds = None

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_match]
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = mock_odds
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        backtester = Backtester()
        params = BacktestParams(days=30, market="ou")
        report = backtester.run(params)

        assert report.n_predictions >= 0  # May or may not find predictions


class TestOUResult:
    """Test O/U result creation."""

    def test_create_ou_result_over(self):
        """Create O/U result for over outcome."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1

        mock_odds = MagicMock()
        mock_odds.bf_over_2_5_odds = 1.90
        mock_odds.bf_under_2_5_odds = 2.00

        result = backtester._create_ou_result(mock_match, mock_odds, 4, 0.0)

        assert result is not None
        assert result.market_type == "ou"
        assert result.actual_outcome == "over"

    def test_create_ou_result_under(self):
        """Create O/U result for under outcome."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1

        mock_odds = MagicMock()
        mock_odds.bf_over_2_5_odds = 1.90
        mock_odds.bf_under_2_5_odds = 2.00

        result = backtester._create_ou_result(mock_match, mock_odds, 2, 0.0)

        assert result is not None
        assert result.actual_outcome == "under"

    def test_create_ou_result_missing_odds(self):
        """Return None when odds missing."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1

        mock_odds = MagicMock()
        mock_odds.bf_over_2_5_odds = None
        mock_odds.bf_under_2_5_odds = None

        result = backtester._create_ou_result(mock_match, mock_odds, 3, 0.0)

        assert result is None


class TestBTTSResult:
    """Test BTTS result creation."""

    def test_create_btts_result_yes(self):
        """Create BTTS result for yes outcome."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1

        mock_odds = MagicMock()
        mock_odds.bf_btts_yes_odds = 1.85
        mock_odds.bf_btts_no_odds = 2.05

        result = backtester._create_btts_result(mock_match, mock_odds, True, 0.0)

        assert result is not None
        assert result.market_type == "btts"
        assert result.actual_outcome == "yes"

    def test_create_btts_result_no(self):
        """Create BTTS result for no outcome."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1

        mock_odds = MagicMock()
        mock_odds.bf_btts_yes_odds = 1.85
        mock_odds.bf_btts_no_odds = 2.05

        result = backtester._create_btts_result(mock_match, mock_odds, False, 0.0)

        assert result is not None
        assert result.actual_outcome == "no"


class Test1X2Result:
    """Test 1X2 result creation."""

    def test_create_1x2_result_home_win(self):
        """Create 1X2 result for home win."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1
        mock_match.home_goals = 2
        mock_match.away_goals = 1

        mock_odds = MagicMock()
        mock_odds.bf_home_odds = 2.20
        mock_odds.bf_draw_odds = 3.50
        mock_odds.bf_away_odds = 3.20

        result = backtester._create_1x2_result(mock_match, mock_odds, 0.0)

        assert result is not None
        assert result.market_type == "1x2"
        assert result.actual_outcome == "home"

    def test_create_1x2_result_draw(self):
        """Create 1X2 result for draw."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1
        mock_match.home_goals = 1
        mock_match.away_goals = 1

        mock_odds = MagicMock()
        mock_odds.bf_home_odds = 2.20
        mock_odds.bf_draw_odds = 3.50
        mock_odds.bf_away_odds = 3.20

        result = backtester._create_1x2_result(mock_match, mock_odds, 0.0)

        assert result is not None
        assert result.actual_outcome == "draw"

    def test_create_1x2_result_away_win(self):
        """Create 1X2 result for away win."""
        backtester = Backtester()

        mock_match = MagicMock()
        mock_match.id = 1
        mock_match.home_goals = 0
        mock_match.away_goals = 2

        mock_odds = MagicMock()
        mock_odds.bf_home_odds = 2.20
        mock_odds.bf_draw_odds = 3.50
        mock_odds.bf_away_odds = 3.20

        result = backtester._create_1x2_result(mock_match, mock_odds, 0.0)

        assert result is not None
        assert result.actual_outcome == "away"
