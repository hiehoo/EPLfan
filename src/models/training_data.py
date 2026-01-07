"""Collect and prepare training data for ML models."""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.storage.database import db
from src.storage.models import Match, TeamStats
from src.models.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example with features and labels."""
    match_id: str
    features: np.ndarray
    feature_names: list[str]

    # Labels (what actually happened)
    label_1x2: str  # "home", "draw", "away"
    label_over_2_5: bool
    label_btts: bool

    # Actual values for verification
    home_goals: int
    away_goals: int
    total_goals: int


class TrainingDataCollector:
    """Collect training data from completed matches."""

    MIN_MATCHES_REQUIRED = 500  # Article recommends 500+
    DATA_DIR = Path("data/training")

    def __init__(self):
        self.feature_builder = FeatureBuilder()
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def collect_from_database(self) -> list[TrainingExample]:
        """Collect training examples from completed matches."""
        examples = []

        with db.session() as session:
            # Get completed matches with results
            matches = session.query(Match).filter(
                Match.is_completed == True,
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            ).all()

            logger.info(f"Found {len(matches)} completed matches")

            for match in matches:
                example = self._create_example(session, match)
                if example:
                    examples.append(example)

        logger.info(f"Created {len(examples)} training examples")
        return examples

    def _create_example(self, session, match: Match) -> Optional[TrainingExample]:
        """Create training example from match."""
        try:
            # Get team stats at match time
            home_stats = session.query(TeamStats).filter(
                TeamStats.team_name == match.home_team,
                TeamStats.snapshot_date <= match.kickoff,
            ).order_by(TeamStats.snapshot_date.desc()).first()

            away_stats = session.query(TeamStats).filter(
                TeamStats.team_name == match.away_team,
                TeamStats.snapshot_date <= match.kickoff,
            ).order_by(TeamStats.snapshot_date.desc()).first()

            if not home_stats or not away_stats:
                return None

            # Build features (convert TeamStats to TeamXGData-like)
            from src.fetchers.understat_fetcher import TeamXGData

            home_xg_data = TeamXGData(
                team_name=home_stats.team_name,
                matches_played=10,
                xg_for=home_stats.xg_for,
                xga_against=home_stats.xga_against,
                xpts=home_stats.xpts,
                home_xg=home_stats.home_xg,
                home_xga=home_stats.home_xga,
                away_xg=home_stats.away_xg,
                away_xga=home_stats.away_xga,
                shots_per_game=home_stats.shots_per_game or 12.0,
                shots_against_per_game=home_stats.shots_against_per_game or 12.0,
                goal_volatility=home_stats.goal_volatility or 1.0,
                shot_conversion_rate=home_stats.shot_conversion_rate or 0.1,
                xg_overperformance=home_stats.xg_overperformance or 0.0,
            )

            away_xg_data = TeamXGData(
                team_name=away_stats.team_name,
                matches_played=10,
                xg_for=away_stats.xg_for,
                xga_against=away_stats.xga_against,
                xpts=away_stats.xpts,
                home_xg=away_stats.home_xg,
                home_xga=away_stats.home_xga,
                away_xg=away_stats.away_xg,
                away_xga=away_stats.away_xga,
                shots_per_game=away_stats.shots_per_game or 12.0,
                shots_against_per_game=away_stats.shots_against_per_game or 12.0,
                goal_volatility=away_stats.goal_volatility or 1.0,
                shot_conversion_rate=away_stats.shot_conversion_rate or 0.1,
                xg_overperformance=away_stats.xg_overperformance or 0.0,
            )

            features = self.feature_builder.build_match_features(
                match_id=str(match.id),
                home_team=match.home_team,
                away_team=match.away_team,
                home_stats=home_xg_data,
                away_stats=away_xg_data,
                home_elo=home_stats.elo_rating,
                away_elo=away_stats.elo_rating,
            )

            # Calculate labels
            total_goals = match.home_goals + match.away_goals
            label_1x2 = "home" if match.home_goals > match.away_goals else (
                "away" if match.away_goals > match.home_goals else "draw"
            )

            return TrainingExample(
                match_id=str(match.id),
                features=features.to_array(),
                feature_names=features.feature_names,
                label_1x2=label_1x2,
                label_over_2_5=total_goals > 2.5,
                label_btts=match.home_goals > 0 and match.away_goals > 0,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                total_goals=total_goals,
            )

        except Exception as e:
            logger.debug(f"Error creating example for match {match.id}: {e}")
            return None

    def save_to_csv(self, examples: list[TrainingExample], filename: str = "training_data.csv"):
        """Save training data to CSV."""
        if not examples:
            logger.warning("No examples to save")
            return

        # Build DataFrame
        data = []
        for ex in examples:
            row = {name: val for name, val in zip(ex.feature_names, ex.features)}
            row.update({
                "match_id": ex.match_id,
                "label_1x2": ex.label_1x2,
                "label_over_2_5": int(ex.label_over_2_5),
                "label_btts": int(ex.label_btts),
                "home_goals": ex.home_goals,
                "away_goals": ex.away_goals,
            })
            data.append(row)

        df = pd.DataFrame(data)
        filepath = self.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} examples to {filepath}")

    def load_from_csv(self, filename: str = "training_data.csv") -> pd.DataFrame:
        """Load training data from CSV."""
        filepath = self.DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()


# Singleton
training_collector = TrainingDataCollector()
