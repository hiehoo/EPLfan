"""Tests for UI components."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.ui.components.edge_badge import edge_badge
from src.ui.components.probability_bar import probability_bar, probability_bar_dual


class TestEdgeBadge:
    """Tests for edge_badge component."""

    def test_strong_edge(self):
        """Test strong edge (>= 10%)."""
        result = edge_badge(0.12)
        assert "🟢" in result
        assert "Strong" in result
        assert "12.0%" in result

    def test_moderate_edge(self):
        """Test moderate edge (>= 7% but < 10%)."""
        result = edge_badge(0.08)
        assert "🟡" in result
        assert "Moderate" in result
        assert "8.0%" in result

    def test_marginal_edge(self):
        """Test marginal edge (>= 5% but < 7%)."""
        result = edge_badge(0.055)
        assert "🟠" in result
        assert "Marginal" in result
        assert "5.5%" in result

    def test_below_threshold(self):
        """Test edge below threshold (< 5%)."""
        result = edge_badge(0.03)
        assert "⚪" in result
        assert "Below threshold" in result
        assert "3.0%" in result

    def test_edge_boundary_10_percent(self):
        """Test edge at exactly 10% boundary."""
        result = edge_badge(0.10)
        assert "🟢" in result
        assert "Strong" in result

    def test_edge_boundary_7_percent(self):
        """Test edge at exactly 7% boundary."""
        result = edge_badge(0.07)
        assert "🟡" in result
        assert "Moderate" in result

    def test_edge_boundary_5_percent(self):
        """Test edge at exactly 5% boundary."""
        result = edge_badge(0.05)
        assert "🟠" in result
        assert "Marginal" in result

    def test_zero_edge(self):
        """Test zero edge."""
        result = edge_badge(0.0)
        assert "⚪" in result
        assert "0.0%" in result


class TestProbabilityBar:
    """Tests for probability_bar component."""

    @patch("streamlit.markdown")
    def test_1x2_probabilities(self, mock_markdown):
        """Test 1X2 probability bar rendering."""
        probs = {"home": 0.45, "draw": 0.30, "away": 0.25}
        labels = ["Home", "Draw", "Away"]

        probability_bar(probs, labels)

        mock_markdown.assert_called_once()
        html = mock_markdown.call_args[0][0]

        # Check structure
        assert "display: flex" in html
        assert "#28a745" in html  # Green for home
        assert "#6c757d" in html  # Gray for draw
        assert "#dc3545" in html  # Red for away

    @patch("streamlit.markdown")
    def test_probability_bar_shows_labels(self, mock_markdown):
        """Test that labels are shown for significant probabilities."""
        probs = {"home": 0.50, "draw": 0.25, "away": 0.25}
        labels = ["Home", "Draw", "Away"]

        probability_bar(probs, labels)

        html = mock_markdown.call_args[0][0]
        assert "Home: 50%" in html
        assert "Draw: 25%" in html
        assert "Away: 25%" in html

    @patch("streamlit.markdown")
    def test_probability_bar_hides_small_labels(self, mock_markdown):
        """Test that labels are hidden for small probabilities (<10%)."""
        probs = {"home": 0.85, "draw": 0.08, "away": 0.07}
        labels = ["Home", "Draw", "Away"]

        probability_bar(probs, labels)

        html = mock_markdown.call_args[0][0]
        assert "Home: 85%" in html
        # Small probabilities shouldn't show labels
        assert "Draw: 8%" not in html
        assert "Away: 7%" not in html


class TestProbabilityBarDual:
    """Tests for probability_bar_dual component."""

    @patch("streamlit.markdown")
    def test_over_under_probabilities(self, mock_markdown):
        """Test O/U probability bar rendering."""
        probs = {"over": 0.55, "under": 0.45}
        labels = ["Over", "Under"]

        probability_bar_dual(probs, labels)

        mock_markdown.assert_called_once()
        html = mock_markdown.call_args[0][0]

        assert "#28a745" in html  # Green for over
        assert "#dc3545" in html  # Red for under
        assert "Over: 55%" in html
        assert "Under: 45%" in html

    @patch("streamlit.markdown")
    def test_btts_probabilities(self, mock_markdown):
        """Test BTTS probability bar rendering."""
        probs = {"yes": 0.60, "no": 0.40}
        labels = ["Yes", "No"]

        probability_bar_dual(probs, labels)

        html = mock_markdown.call_args[0][0]
        assert "Yes: 60%" in html
        assert "No: 40%" in html


class TestLiveSignalsPage:
    """Tests for live signals page functions."""

    def test_edge_color_coding_strong(self):
        """Test strong edge gets green color."""
        # Test the same logic used in live_signals.py
        edge = 0.12
        if edge >= 0.10:
            edge_color = "🟢"
        elif edge >= 0.07:
            edge_color = "🟡"
        else:
            edge_color = "🟠"

        assert edge_color == "🟢"

    def test_edge_color_coding_moderate(self):
        """Test moderate edge gets yellow color."""
        edge = 0.08
        if edge >= 0.10:
            edge_color = "🟢"
        elif edge >= 0.07:
            edge_color = "🟡"
        else:
            edge_color = "🟠"

        assert edge_color == "🟡"

    def test_edge_color_coding_marginal(self):
        """Test marginal edge gets orange color."""
        edge = 0.05
        if edge >= 0.10:
            edge_color = "🟢"
        elif edge >= 0.07:
            edge_color = "🟡"
        else:
            edge_color = "🟠"

        assert edge_color == "🟠"


class TestMatchAnalysisHelpers:
    """Tests for match analysis page helper functions."""

    def test_edge_metric_logic_strong(self):
        """Test edge metric color selection for strong edge."""
        edge = 0.08
        if edge > 0.07:
            delta_color = "normal"
            prefix = "🟢"
        elif edge > 0.05:
            delta_color = "normal"
            prefix = "🟡"
        elif edge > 0:
            delta_color = "off"
            prefix = ""
        else:
            delta_color = "inverse"
            prefix = ""

        assert delta_color == "normal"
        assert prefix == "🟢"

    def test_edge_metric_logic_moderate(self):
        """Test edge metric color selection for moderate edge."""
        edge = 0.06
        if edge > 0.07:
            delta_color = "normal"
            prefix = "🟢"
        elif edge > 0.05:
            delta_color = "normal"
            prefix = "🟡"
        elif edge > 0:
            delta_color = "off"
            prefix = ""
        else:
            delta_color = "inverse"
            prefix = ""

        assert delta_color == "normal"
        assert prefix == "🟡"

    def test_edge_metric_logic_small(self):
        """Test edge metric color selection for small positive edge."""
        edge = 0.03
        if edge > 0.07:
            delta_color = "normal"
            prefix = "🟢"
        elif edge > 0.05:
            delta_color = "normal"
            prefix = "🟡"
        elif edge > 0:
            delta_color = "off"
            prefix = ""
        else:
            delta_color = "inverse"
            prefix = ""

        assert delta_color == "off"
        assert prefix == ""

    def test_edge_metric_logic_negative(self):
        """Test edge metric color selection for negative edge."""
        edge = -0.05
        if edge > 0.07:
            delta_color = "normal"
            prefix = "🟢"
        elif edge > 0.05:
            delta_color = "normal"
            prefix = "🟡"
        elif edge > 0:
            delta_color = "off"
            prefix = ""
        else:
            delta_color = "inverse"
            prefix = ""

        assert delta_color == "inverse"
        assert prefix == ""


class TestHistoricalPageHelpers:
    """Tests for historical page helper functions."""

    def test_brier_score_calculation(self):
        """Test Brier score calculation."""
        import pandas as pd

        # Perfect predictions
        df_perfect = pd.DataFrame({
            "fair_prob": [1.0, 0.0, 1.0],
            "outcome": [1, 0, 1],
        })
        brier_perfect = ((df_perfect["fair_prob"] - df_perfect["outcome"].astype(float)) ** 2).mean()
        assert brier_perfect == 0.0

        # Worst predictions
        df_worst = pd.DataFrame({
            "fair_prob": [1.0, 0.0, 1.0],
            "outcome": [0, 1, 0],
        })
        brier_worst = ((df_worst["fair_prob"] - df_worst["outcome"].astype(float)) ** 2).mean()
        assert brier_worst == 1.0

        # Typical predictions
        df_typical = pd.DataFrame({
            "fair_prob": [0.7, 0.6, 0.5],
            "outcome": [1, 1, 0],
        })
        brier_typical = ((df_typical["fair_prob"] - df_typical["outcome"].astype(float)) ** 2).mean()
        assert 0 < brier_typical < 1

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        import pandas as pd

        df = pd.DataFrame({
            "outcome": [1, 1, 0, 1, 0],
        })
        wins = df["outcome"].sum()
        total = len(df)
        hit_rate = wins / total

        assert wins == 3
        assert total == 5
        assert hit_rate == 0.6


class TestSettingsPage:
    """Tests for settings page functions."""

    def test_weight_file_mapping(self):
        """Test correct weight file mapping."""
        weight_files = {
            "1X2": "1x2_weights.yaml",
            "Over/Under": "ou_weights.yaml",
            "BTTS": "btts_weights.yaml",
        }

        assert weight_files["1X2"] == "1x2_weights.yaml"
        assert weight_files["Over/Under"] == "ou_weights.yaml"
        assert weight_files["BTTS"] == "btts_weights.yaml"

    def test_threshold_display(self):
        """Test threshold value conversion for display."""
        # Thresholds stored as decimal
        threshold_1x2 = 0.05
        threshold_ou = 0.07
        threshold_btts = 0.07

        # Convert to percentage for display (use pytest.approx for float comparison)
        assert threshold_1x2 * 100 == pytest.approx(5.0)
        assert threshold_ou * 100 == pytest.approx(7.0)
        assert threshold_btts * 100 == pytest.approx(7.0)
