"""Tests for feature builder and rolling windows."""
import pytest
import numpy as np
from datetime import datetime, timezone

from src.models.feature_builder import FeatureBuilder, MatchFeatures
from src.fetchers.understat_fetcher import TeamXGData
from src.config.settings import load_window_config, WindowConfig


class TestMatchFeatures:
    """Tests for MatchFeatures dataclass."""

    def test_to_array_shape(self):
        """Feature array should have correct shape."""
        features = MatchFeatures(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_shots_pg=15.0,
            away_shots_pg=12.0,
        )

        arr = features.to_array()
        assert arr.shape == (14,)  # 14 features

    def test_feature_names_match_array(self):
        """Feature names should match array length."""
        features = MatchFeatures(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
        )

        assert len(features.feature_names) == len(features.to_array())

    def test_to_array_values(self):
        """Feature array should contain correct values."""
        features = MatchFeatures(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_shots_pg=15.0,
            away_shots_pg=12.0,
            shot_ratio=1.25,
            home_xg=2.0,
            away_xg=1.5,
            xg_diff=0.5,
            combined_volatility=1.0,
            home_form_xg=1.8,
            away_form_xg=1.4,
            home_conversion=0.12,
            away_conversion=0.10,
            elo_diff=100,
            bf_over_2_5_implied=0.55,
            is_elite_match=True,
        )

        arr = features.to_array()
        assert arr[0] == 15.0  # home_shots_pg
        assert arr[1] == 12.0  # away_shots_pg
        assert arr[2] == 1.25  # shot_ratio
        assert arr[13] == 1.0  # is_elite_match (True -> 1.0)

    def test_non_elite_match(self):
        """Non-elite match should have 0.0 for is_elite feature."""
        features = MatchFeatures(
            match_id="1",
            home_team="Burnley",
            away_team="Southampton",
            is_elite_match=False,
        )

        arr = features.to_array()
        assert arr[13] == 0.0  # is_elite_match = False


class TestFeatureBuilder:
    """Tests for FeatureBuilder."""

    @pytest.fixture
    def home_stats(self):
        return TeamXGData(
            team_name="Arsenal",
            matches_played=10,
            xg_for=1.8,
            xga_against=1.0,
            xpts=2.2,
            home_xg=2.0,
            home_xga=0.8,
            away_xg=1.6,
            away_xga=1.2,
            shots_per_game=15.0,
            shots_against_per_game=10.0,
            goal_volatility=1.2,
            shot_conversion_rate=0.12,
            xg_overperformance=0.1,
            form_xg=[1.5, 2.0, 1.8, 2.2, 1.9],
        )

    @pytest.fixture
    def away_stats(self):
        return TeamXGData(
            team_name="Chelsea",
            matches_played=10,
            xg_for=1.5,
            xga_against=1.2,
            xpts=1.8,
            home_xg=1.8,
            home_xga=1.0,
            away_xg=1.2,
            away_xga=1.4,
            shots_per_game=12.0,
            shots_against_per_game=11.0,
            goal_volatility=0.9,
            shot_conversion_rate=0.10,
            xg_overperformance=-0.05,
            form_xg=[1.3, 1.5, 1.2, 1.6, 1.4],
        )

    def test_build_features_shot_ratio(self, home_stats, away_stats):
        """Shot ratio should be calculated correctly."""
        builder = FeatureBuilder()

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        expected_ratio = 15.0 / 12.0
        assert abs(features.shot_ratio - expected_ratio) < 0.01

    def test_build_features_xg_diff(self, home_stats, away_stats):
        """xG diff should be home attack - away defense."""
        builder = FeatureBuilder()

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        # home_xg_for - away_xga_against
        expected = home_stats.xg_for - away_stats.xga_against
        assert abs(features.xg_diff - expected) < 0.01

    def test_build_features_combined_volatility(self, home_stats, away_stats):
        """Combined volatility should be average of both teams."""
        builder = FeatureBuilder()

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        expected = (home_stats.goal_volatility + away_stats.goal_volatility) / 2
        assert abs(features.combined_volatility - expected) < 0.01

    def test_build_features_form_xg(self, home_stats, away_stats):
        """Form xG should be average of recent matches."""
        builder = FeatureBuilder()

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        expected_home_form = sum(home_stats.form_xg) / len(home_stats.form_xg)
        expected_away_form = sum(away_stats.form_xg) / len(away_stats.form_xg)

        assert abs(features.home_form_xg - expected_home_form) < 0.01
        assert abs(features.away_form_xg - expected_away_form) < 0.01

    def test_elite_match_detection(self, home_stats, away_stats):
        """Elite matches should be detected."""
        builder = FeatureBuilder()

        # Manchester City is elite
        features_elite = builder.build_match_features(
            match_id="1",
            home_team="Manchester City",
            away_team="Burnley",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        features_normal = builder.build_match_features(
            match_id="2",
            home_team="Burnley",
            away_team="Southampton",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        assert features_elite.is_elite_match
        assert not features_normal.is_elite_match

    def test_build_features_with_bf_odds(self, home_stats, away_stats):
        """Features should incorporate Betfair odds when provided."""
        builder = FeatureBuilder()

        bf_odds = {
            "home": 2.0,  # 50% implied
            "away": 4.0,  # 25% implied
            "over_2_5": 1.8,  # ~55.6% implied
        }

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
            bf_odds=bf_odds,
        )

        assert abs(features.bf_home_implied - 0.5) < 0.01
        assert abs(features.bf_away_implied - 0.25) < 0.01
        assert abs(features.bf_over_2_5_implied - (1/1.8)) < 0.01

    def test_build_features_elo_diff(self, home_stats, away_stats):
        """ELO diff should be home - away."""
        builder = FeatureBuilder()

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
            home_elo=1900,
            away_elo=1850,
        )

        assert features.elo_diff == 50
        assert features.home_elo == 1900
        assert features.away_elo == 1850

    def test_shot_ratio_with_zero_away_shots(self, home_stats, away_stats):
        """Shot ratio should default to 1.0 when away shots is 0."""
        builder = FeatureBuilder()

        away_stats.shots_per_game = 0.0

        features = builder.build_match_features(
            match_id="1",
            home_team="Arsenal",
            away_team="Chelsea",
            home_stats=home_stats,
            away_stats=away_stats,
        )

        assert features.shot_ratio == 1.0


class TestWindowConfig:
    """Tests for rolling window configuration."""

    def test_window_config_defaults(self):
        """WindowConfig should have sensible defaults."""
        config = WindowConfig()

        assert config.window_1x2 == 10
        assert config.window_ou_2_5 == 6
        assert config.window_btts == 4
        assert config.window_shots == 5
        assert config.window_xg == 8

    def test_load_window_config(self):
        """load_window_config should return dict."""
        config = load_window_config()

        assert isinstance(config, dict)
        # Should have rolling_windows key if file exists
        if config:
            assert "rolling_windows" in config or config == {}

    def test_feature_builder_window_size(self):
        """FeatureBuilder should return correct window sizes."""
        builder = FeatureBuilder()

        # Test 1x2 window
        window_1x2 = builder.get_window_size("1x2")
        assert window_1x2 == 10  # default

        # Test O/U 2.5 window
        window_ou = builder.get_window_size("ou", line=2.5)
        assert window_ou == 6  # default

        # Test BTTS window
        window_btts = builder.get_window_size("btts")
        assert window_btts == 4  # default


class TestEnhancedTeamXGData:
    """Tests for enhanced TeamXGData with shot/volatility fields."""

    def test_team_xg_data_has_shot_fields(self):
        """TeamXGData should include shot metrics."""
        data = TeamXGData(
            team_name="Arsenal",
            matches_played=10,
            xg_for=1.8,
            xga_against=1.0,
            xpts=2.2,
            home_xg=2.0,
            home_xga=0.8,
            away_xg=1.6,
            away_xga=1.2,
            shots_per_game=15.0,
            shots_against_per_game=10.0,
            goal_volatility=1.2,
        )

        assert data.shots_per_game == 15.0
        assert data.shots_against_per_game == 10.0
        assert data.goal_volatility == 1.2

    def test_team_xg_data_has_volatility_fields(self):
        """TeamXGData should include volatility metrics."""
        data = TeamXGData(
            team_name="Arsenal",
            matches_played=10,
            xg_for=1.8,
            xga_against=1.0,
            xpts=2.2,
            home_xg=2.0,
            home_xga=0.8,
            away_xg=1.6,
            away_xga=1.2,
            goal_volatility=1.2,
            goals_against_volatility=0.9,
            xg_volatility=0.5,
        )

        assert data.goal_volatility == 1.2
        assert data.goals_against_volatility == 0.9
        assert data.xg_volatility == 0.5

    def test_team_xg_data_has_conversion_fields(self):
        """TeamXGData should include conversion/efficiency metrics."""
        data = TeamXGData(
            team_name="Arsenal",
            matches_played=10,
            xg_for=1.8,
            xga_against=1.0,
            xpts=2.2,
            home_xg=2.0,
            home_xga=0.8,
            away_xg=1.6,
            away_xga=1.2,
            shot_conversion_rate=0.12,
            xg_overperformance=0.15,
        )

        assert data.shot_conversion_rate == 0.12
        assert data.xg_overperformance == 0.15

    def test_team_xg_data_defaults(self):
        """TeamXGData should have sensible defaults for new fields."""
        data = TeamXGData(
            team_name="Arsenal",
            matches_played=10,
            xg_for=1.8,
            xga_against=1.0,
            xpts=2.2,
            home_xg=2.0,
            home_xga=0.8,
            away_xg=1.6,
            away_xga=1.2,
        )

        assert data.shots_per_game == 0.0
        assert data.goal_volatility == 0.0
        assert data.shot_conversion_rate == 0.0
        assert data.xg_overperformance == 0.0
