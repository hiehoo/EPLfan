"""Tests for historical data seeder."""
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.fetchers.historical_data_seeder import (
    HistoricalDataSeeder,
    TEAM_NAME_MAP,
)


class TestTeamNameNormalization:
    """Test team name normalization."""

    def test_normalize_known_team(self):
        """Normalize known team names."""
        seeder = HistoricalDataSeeder()
        assert seeder._normalize_team_name("Man United") == "Manchester United"
        assert seeder._normalize_team_name("Man City") == "Manchester City"
        assert seeder._normalize_team_name("Nott'm Forest") == "Nottingham Forest"
        assert seeder._normalize_team_name("Spurs") == "Tottenham"
        assert seeder._normalize_team_name("Wolves") == "Wolverhampton"

    def test_normalize_unknown_team_passthrough(self):
        """Unknown team names pass through unchanged."""
        seeder = HistoricalDataSeeder()
        assert seeder._normalize_team_name("Arsenal") == "Arsenal"
        assert seeder._normalize_team_name("Liverpool") == "Liverpool"
        assert seeder._normalize_team_name("Chelsea") == "Chelsea"

    def test_team_name_map_coverage(self):
        """Verify team name map has expected entries."""
        assert "Man United" in TEAM_NAME_MAP
        assert "Man City" in TEAM_NAME_MAP
        assert "Nott'm Forest" in TEAM_NAME_MAP


class TestOddsConversion:
    """Test odds to probability conversion."""

    def test_odds_to_prob_standard(self):
        """Convert standard odds to probability."""
        seeder = HistoricalDataSeeder()
        assert abs(seeder._odds_to_prob(2.0) - 0.5) < 0.001
        assert abs(seeder._odds_to_prob(4.0) - 0.25) < 0.001
        assert abs(seeder._odds_to_prob(1.5) - 0.667) < 0.01

    def test_odds_to_prob_edge_cases(self):
        """Handle edge case odds."""
        seeder = HistoricalDataSeeder()
        assert seeder._odds_to_prob(1.0) == 0.0  # 1.0 odds = invalid
        assert seeder._odds_to_prob(0.5) == 0.0  # below 1.0 = invalid


class TestSafeFloat:
    """Test safe float conversion."""

    def test_safe_float_valid(self):
        """Convert valid float values."""
        seeder = HistoricalDataSeeder()
        assert seeder._safe_float(2.5) == 2.5
        assert seeder._safe_float("3.0") == 3.0
        assert seeder._safe_float(1) == 1.0

    def test_safe_float_invalid(self):
        """Handle invalid values."""
        seeder = HistoricalDataSeeder()
        assert seeder._safe_float(None) is None
        assert seeder._safe_float(pd.NA) is None
        assert seeder._safe_float("invalid") is None


class TestCSVDownload:
    """Test CSV download functionality."""

    @patch("src.fetchers.historical_data_seeder.requests.get")
    def test_download_csv_success(self, mock_get):
        """Successfully download and parse CSV."""
        csv_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A
12/08/2023,Arsenal,Liverpool,2,1,2.50,3.40,2.90
19/08/2023,Chelsea,Man United,0,3,2.00,3.50,4.00"""

        mock_response = MagicMock()
        mock_response.text = csv_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        seeder = HistoricalDataSeeder()
        df = seeder._download_csv("2324")

        assert df is not None
        assert len(df) == 2
        assert "HomeTeam" in df.columns
        assert "FTHG" in df.columns

    @patch("src.fetchers.historical_data_seeder.requests.get")
    def test_download_csv_missing_columns(self, mock_get):
        """Reject CSV with missing required columns."""
        csv_content = """Date,HomeTeam,AwayTeam
12/08/2023,Arsenal,Liverpool"""

        mock_response = MagicMock()
        mock_response.text = csv_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        seeder = HistoricalDataSeeder()
        df = seeder._download_csv("2324")

        assert df is None

    @patch("src.fetchers.historical_data_seeder.requests.get")
    def test_download_csv_network_error(self, mock_get):
        """Handle network errors gracefully."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        seeder = HistoricalDataSeeder()
        df = seeder._download_csv("2324")

        assert df is None


class TestMatchImport:
    """Test match import functionality."""

    @patch("src.fetchers.historical_data_seeder.db")
    def test_import_match_creates_records(self, mock_db):
        """Import match creates Match and OddsSnapshot."""
        # Setup mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        row = pd.Series({
            "Date": "12/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
            "B365H": 2.50,
            "B365D": 3.40,
            "B365A": 2.90,
            "B365>2.5": 1.80,
            "B365<2.5": 2.10,
        })

        seeder = HistoricalDataSeeder()
        result = seeder._import_match(row, "2324")

        assert result is True
        assert mock_session.add.call_count == 2  # Match + OddsSnapshot

    @patch("src.fetchers.historical_data_seeder.db")
    def test_import_match_skips_duplicate(self, mock_db):
        """Skip duplicate matches."""
        # Setup mock with existing match
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        row = pd.Series({
            "Date": "12/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
        })

        seeder = HistoricalDataSeeder()
        result = seeder._import_match(row, "2324")

        assert result is False


class TestSeasonSeeding:
    """Test season seeding workflow."""

    @patch.object(HistoricalDataSeeder, "_download_csv")
    @patch.object(HistoricalDataSeeder, "_import_match")
    def test_seed_season_counts(self, mock_import, mock_download):
        """Track imported vs skipped counts."""
        mock_download.return_value = pd.DataFrame({
            "Date": ["12/08/2023", "19/08/2023", "26/08/2023"],
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Liverpool", "Man United", "Arsenal"],
            "FTHG": [2, 0, 1],
            "FTAG": [1, 3, 1],
        })

        # 2 imported, 1 skipped
        mock_import.side_effect = [True, True, False]

        seeder = HistoricalDataSeeder()
        result = seeder.seed_season("2324")

        assert result["total"] == 3
        assert result["imported"] == 2
        assert result["skipped"] == 1

    @patch.object(HistoricalDataSeeder, "_download_csv")
    def test_seed_season_empty_csv(self, mock_download):
        """Handle empty CSV gracefully."""
        mock_download.return_value = pd.DataFrame()

        seeder = HistoricalDataSeeder()
        result = seeder.seed_season("2324")

        assert result["total"] == 0
        assert result["imported"] == 0


class TestDateParsing:
    """Test date parsing formats."""

    @patch("src.fetchers.historical_data_seeder.db")
    def test_parse_date_yyyy_format(self, mock_db):
        """Parse dd/mm/yyyy date format."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        row = pd.Series({
            "Date": "12/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
        })

        seeder = HistoricalDataSeeder()
        result = seeder._import_match(row, "2324")
        assert result is True

    @patch("src.fetchers.historical_data_seeder.db")
    def test_parse_date_yy_format(self, mock_db):
        """Parse dd/mm/yy date format."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=False)

        row = pd.Series({
            "Date": "12/08/23",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Liverpool",
            "FTHG": 2,
            "FTAG": 1,
        })

        seeder = HistoricalDataSeeder()
        result = seeder._import_match(row, "2324")
        assert result is True
