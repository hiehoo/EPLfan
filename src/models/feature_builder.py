"""Build features for ML classification model."""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config.settings import load_window_config
from src.fetchers.understat_fetcher import TeamXGData

logger = logging.getLogger(__name__)


@dataclass
class MatchFeatures:
    """Feature vector for a single match."""
    match_id: str
    home_team: str
    away_team: str

    # Shot features (top importance per article)
    home_shots_pg: float = 0.0
    away_shots_pg: float = 0.0
    home_shots_against: float = 0.0
    away_shots_against: float = 0.0
    shot_ratio: float = 1.0  # home_shots / away_shots

    # xG features
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_xga: float = 0.0
    away_xga: float = 0.0
    xg_diff: float = 0.0  # home_xg - away_xga (attacking strength)

    # Volatility features
    home_goal_volatility: float = 0.0
    away_goal_volatility: float = 0.0
    combined_volatility: float = 0.0

    # Form features
    home_form_xg: float = 0.0  # Last 5 avg xG
    away_form_xg: float = 0.0
    home_form_trend: float = 0.0  # xG slope (improving/declining)
    away_form_trend: float = 0.0

    # Efficiency features
    home_conversion: float = 0.0
    away_conversion: float = 0.0
    home_xg_overperf: float = 0.0
    away_xg_overperf: float = 0.0

    # ELO
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    elo_diff: float = 0.0

    # Market features
    bf_home_implied: float = 0.33
    bf_away_implied: float = 0.33
    bf_over_2_5_implied: float = 0.5

    # Meta features
    is_elite_match: bool = False
    hours_to_kickoff: float = 48.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            self.home_shots_pg,
            self.away_shots_pg,
            self.shot_ratio,
            self.home_xg,
            self.away_xg,
            self.xg_diff,
            self.combined_volatility,
            self.home_form_xg,
            self.away_form_xg,
            self.home_conversion,
            self.away_conversion,
            self.elo_diff,
            self.bf_over_2_5_implied,
            1.0 if self.is_elite_match else 0.0,
        ])

    @property
    def feature_names(self) -> list[str]:
        """Feature names for model interpretation."""
        return [
            "home_shots_pg",
            "away_shots_pg",
            "shot_ratio",
            "home_xg",
            "away_xg",
            "xg_diff",
            "combined_volatility",
            "home_form_xg",
            "away_form_xg",
            "home_conversion",
            "away_conversion",
            "elo_diff",
            "bf_over_2_5_implied",
            "is_elite_match",
        ]


class FeatureBuilder:
    """Build feature vectors for ML model."""

    def __init__(self):
        self.window_config = load_window_config()

    def build_match_features(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        home_stats: TeamXGData,
        away_stats: TeamXGData,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
        bf_odds: Optional[dict] = None,
        hours_to_kickoff: float = 48.0,
    ) -> MatchFeatures:
        """Build feature vector for a match."""
        from src.models.base_rate_tracker import base_rate_tracker

        # Shot features
        shot_ratio = (
            home_stats.shots_per_game / away_stats.shots_per_game
            if away_stats.shots_per_game > 0 else 1.0
        )

        # xG diff (attacking potential)
        xg_diff = home_stats.xg_for - away_stats.xga_against

        # Combined volatility
        combined_vol = (
            home_stats.goal_volatility + away_stats.goal_volatility
        ) / 2

        # Form trend (simplified: compare recent to season avg)
        home_form_xg = (
            sum(home_stats.form_xg) / len(home_stats.form_xg)
            if home_stats.form_xg else home_stats.xg_for
        )
        away_form_xg = (
            sum(away_stats.form_xg) / len(away_stats.form_xg)
            if away_stats.form_xg else away_stats.xg_for
        )

        # Form trend: positive = improving
        home_form_trend = home_form_xg - home_stats.xg_for
        away_form_trend = away_form_xg - away_stats.xg_for

        # Market implied (if available)
        bf_home = 0.33
        bf_away = 0.33
        bf_over = 0.5
        if bf_odds:
            bf_home = 1 / bf_odds.get("home", 3.0) if bf_odds.get("home") else 0.33
            bf_away = 1 / bf_odds.get("away", 3.0) if bf_odds.get("away") else 0.33
            bf_over = 1 / bf_odds.get("over_2_5", 2.0) if bf_odds.get("over_2_5") else 0.5

        # Elite check
        is_elite = (
            base_rate_tracker.is_elite_team(home_team) or
            base_rate_tracker.is_elite_team(away_team)
        )

        return MatchFeatures(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_shots_pg=home_stats.shots_per_game,
            away_shots_pg=away_stats.shots_per_game,
            home_shots_against=home_stats.shots_against_per_game,
            away_shots_against=away_stats.shots_against_per_game,
            shot_ratio=shot_ratio,
            home_xg=home_stats.home_xg,
            away_xg=away_stats.away_xg,
            home_xga=home_stats.home_xga,
            away_xga=away_stats.away_xga,
            xg_diff=xg_diff,
            home_goal_volatility=home_stats.goal_volatility,
            away_goal_volatility=away_stats.goal_volatility,
            combined_volatility=combined_vol,
            home_form_xg=home_form_xg,
            away_form_xg=away_form_xg,
            home_form_trend=home_form_trend,
            away_form_trend=away_form_trend,
            home_conversion=home_stats.shot_conversion_rate,
            away_conversion=away_stats.shot_conversion_rate,
            home_xg_overperf=home_stats.xg_overperformance,
            away_xg_overperf=away_stats.xg_overperformance,
            home_elo=home_elo,
            away_elo=away_elo,
            elo_diff=home_elo - away_elo,
            bf_home_implied=bf_home,
            bf_away_implied=bf_away,
            bf_over_2_5_implied=bf_over,
            is_elite_match=is_elite,
            hours_to_kickoff=hours_to_kickoff,
        )

    def get_window_size(self, market_type: str, line: Optional[float] = None) -> int:
        """Get rolling window size for market type."""
        windows = self.window_config.get("rolling_windows", {})

        if market_type == "1x2":
            return windows.get("1x2", {}).get("default", 10)
        elif market_type == "ou":
            ou_config = windows.get("ou", {})
            if line == 2.5:
                return ou_config.get("line_2_5", {}).get("default", 6)
            elif line == 1.5:
                return ou_config.get("line_1_5", {}).get("default", 4)
            elif line == 3.5:
                return ou_config.get("line_3_5", {}).get("default", 8)
            return 6
        elif market_type == "btts":
            return windows.get("btts", {}).get("default", 4)

        return 10  # Default


# Singleton
feature_builder = FeatureBuilder()
