"""Tests for storage layer (database and models)."""
import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from src.storage.models import Base, Match, OddsSnapshot, TeamStats, Prediction
from src.storage.database import Database


class TestDatabase:
    """Test database connection and operations."""

    def test_database_creates_tables(self, tmp_path):
        """Test database initialization creates all tables."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        assert db_path.exists()

    def test_session_context_manager(self, tmp_path):
        """Test session context manager commits and closes properly."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        # Add a match using session
        with database.session() as session:
            match = Match(
                external_id="test-123",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff=datetime.now(timezone.utc),
            )
            session.add(match)

        # Verify it was committed
        with database.session() as session:
            result = session.query(Match).filter_by(external_id="test-123").first()
            assert result is not None
            assert result.home_team == "Arsenal"

    def test_session_rollback_on_error(self, tmp_path):
        """Test session rollback on exception."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        # Add a match then try to cause error
        with database.session() as session:
            match = Match(
                external_id="unique-id",
                home_team="Liverpool",
                away_team="Everton",
                kickoff=datetime.now(timezone.utc),
            )
            session.add(match)

        # Try to add duplicate (should fail due to unique constraint)
        with pytest.raises(Exception):
            with database.session() as session:
                duplicate = Match(
                    external_id="unique-id",  # Same ID
                    home_team="City",
                    away_team="United",
                    kickoff=datetime.now(timezone.utc),
                )
                session.add(duplicate)


class TestMatchModel:
    """Test Match model."""

    def test_match_creation(self, tmp_path):
        """Test creating a match record."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-12345",
                home_team="Manchester City",
                away_team="Manchester United",
                kickoff=datetime(2026, 1, 10, 15, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            assert match.id is not None
            assert match.is_completed is False
            assert match.home_goals is None

    def test_match_with_result(self, tmp_path):
        """Test match with final result."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-completed",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
                home_goals=2,
                away_goals=1,
                is_completed=True,
            )
            session.add(match)

        with database.session() as session:
            result = session.query(Match).filter_by(external_id="bf-completed").first()
            assert result.home_goals == 2
            assert result.away_goals == 1
            assert result.is_completed is True


class TestOddsSnapshotModel:
    """Test OddsSnapshot model."""

    def test_odds_snapshot_creation(self, tmp_path):
        """Test creating odds snapshot linked to match."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-odds-test",
                home_team="Liverpool",
                away_team="Tottenham",
                kickoff=datetime(2026, 1, 15, 17, 30, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            snapshot = OddsSnapshot(
                match_id=match.id,
                snapshot_time=datetime.now(timezone.utc),
                hours_to_kickoff=48.5,
                bf_home_odds=1.75,
                bf_draw_odds=3.60,
                bf_away_odds=4.50,
                bf_over_2_5_odds=1.90,
                bf_under_2_5_odds=2.00,
            )
            session.add(snapshot)

        with database.session() as session:
            result = session.query(OddsSnapshot).first()
            assert result.bf_home_odds == 1.75
            assert result.bf_draw_odds == 3.60
            assert result.hours_to_kickoff == 48.5

    def test_odds_snapshot_polymarket_prices(self, tmp_path):
        """Test snapshot with Polymarket prices."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-poly-test",
                home_team="Newcastle",
                away_team="Brighton",
                kickoff=datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            snapshot = OddsSnapshot(
                match_id=match.id,
                snapshot_time=datetime.now(timezone.utc),
                hours_to_kickoff=24.0,
                bf_home_odds=2.10,
                bf_draw_odds=3.40,
                bf_away_odds=3.50,
                pm_home_price=0.48,
                pm_draw_price=0.28,
                pm_away_price=0.24,
                pm_home_liquidity=5000.0,
            )
            session.add(snapshot)

        with database.session() as session:
            result = session.query(OddsSnapshot).first()
            assert result.pm_home_price == 0.48
            assert result.pm_home_liquidity == 5000.0


class TestTeamStatsModel:
    """Test TeamStats model."""

    def test_team_stats_creation(self, tmp_path):
        """Test creating team statistics record."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            stats = TeamStats(
                team_name="Arsenal",
                snapshot_date=datetime.now(timezone.utc),
                xg_for=1.85,
                xga_against=0.95,
                xpts=2.15,
                home_xg=2.10,
                away_xg=1.60,
                home_xga=0.80,
                away_xga=1.10,
                elo_rating=1920.5,
                elo_rank=2,
            )
            session.add(stats)

        with database.session() as session:
            result = session.query(TeamStats).filter_by(team_name="Arsenal").first()
            assert result.xg_for == 1.85
            assert result.elo_rating == 1920.5
            assert result.elo_rank == 2


class TestPredictionModel:
    """Test Prediction model."""

    def test_prediction_creation(self, tmp_path):
        """Test creating prediction record."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-pred-test",
                home_team="Chelsea",
                away_team="West Ham",
                kickoff=datetime(2026, 1, 25, 15, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            prediction = Prediction(
                match_id=match.id,
                preset_used="balanced",
                p_home_model=0.55,
                p_draw_model=0.25,
                p_away_model=0.20,
                p_home_market=0.50,
                p_draw_market=0.28,
                p_away_market=0.22,
                edge_1x2=0.05,
                signal_1x2="home",
                p_over_2_5_model=0.58,
                p_over_2_5_market=0.52,
                edge_ou=0.06,
                signal_ou=None,  # Below threshold
                p_btts_yes_model=0.62,
                p_btts_yes_market=0.55,
                edge_btts=0.07,
                signal_btts="yes",
            )
            session.add(prediction)

        with database.session() as session:
            result = session.query(Prediction).first()
            assert result.preset_used == "balanced"
            assert result.signal_1x2 == "home"
            assert result.edge_1x2 == 0.05
            assert result.signal_ou is None  # No signal below threshold

    def test_prediction_outcome_tracking(self, tmp_path):
        """Test updating prediction with actual outcomes."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-outcome-test",
                home_team="Everton",
                away_team="Fulham",
                kickoff=datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            prediction = Prediction(
                match_id=match.id,
                preset_used="market_trust",
                p_home_model=0.45,
                p_draw_model=0.30,
                p_away_model=0.25,
                p_home_market=0.42,
                p_draw_market=0.32,
                p_away_market=0.26,
                edge_1x2=0.03,
                signal_1x2=None,
                p_over_2_5_model=0.48,
                p_over_2_5_market=0.50,
                edge_ou=-0.02,
                signal_ou=None,
                p_btts_yes_model=0.55,
                p_btts_yes_market=0.52,
                edge_btts=0.03,
                signal_btts=None,
            )
            session.add(prediction)

        # Simulate post-match update
        with database.session() as session:
            pred = session.query(Prediction).first()
            pred.is_1x2_correct = True  # Home won as predicted
            pred.is_ou_correct = False
            pred.is_btts_correct = True

        with database.session() as session:
            result = session.query(Prediction).first()
            assert result.is_1x2_correct is True
            assert result.is_ou_correct is False
            assert result.is_btts_correct is True


class TestMatchRelationships:
    """Test relationships between models."""

    def test_match_odds_snapshots_relationship(self, tmp_path):
        """Test Match to OddsSnapshot relationship."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-rel-test",
                home_team="Aston Villa",
                away_team="Brentford",
                kickoff=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            # Add multiple snapshots
            for hours in [72, 48, 24, 12]:
                snapshot = OddsSnapshot(
                    match_id=match.id,
                    snapshot_time=datetime.now(timezone.utc),
                    hours_to_kickoff=float(hours),
                    bf_home_odds=2.0 + (hours / 100),
                    bf_draw_odds=3.5,
                    bf_away_odds=3.8,
                )
                session.add(snapshot)

        with database.session() as session:
            match = session.query(Match).filter_by(external_id="bf-rel-test").first()
            assert len(match.odds_snapshots) == 4

    def test_match_predictions_relationship(self, tmp_path):
        """Test Match to Prediction relationship."""
        db_path = tmp_path / "test.db"
        database = Database(db_path=db_path)
        database.create_tables()

        with database.session() as session:
            match = Match(
                external_id="bf-pred-rel",
                home_team="Wolves",
                away_team="Leicester",
                kickoff=datetime(2026, 2, 5, 20, 0, tzinfo=timezone.utc),
            )
            session.add(match)
            session.flush()

            # Predictions at different time presets
            for preset in ["analytics_first", "balanced", "market_trust"]:
                pred = Prediction(
                    match_id=match.id,
                    preset_used=preset,
                    p_home_model=0.45,
                    p_draw_model=0.30,
                    p_away_model=0.25,
                    p_home_market=0.42,
                    p_draw_market=0.32,
                    p_away_market=0.26,
                    edge_1x2=0.03,
                    p_over_2_5_model=0.50,
                    p_over_2_5_market=0.48,
                    edge_ou=0.02,
                    p_btts_yes_model=0.55,
                    p_btts_yes_market=0.52,
                    edge_btts=0.03,
                )
                session.add(pred)

        with database.session() as session:
            match = session.query(Match).filter_by(external_id="bf-pred-rel").first()
            assert len(match.predictions) == 3
            presets = [p.preset_used for p in match.predictions]
            assert "analytics_first" in presets
            assert "balanced" in presets
            assert "market_trust" in presets
