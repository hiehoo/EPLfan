"""Tests for data fetchers."""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from src.fetchers.betfair_fetcher import BetfairFetcher, BetfairOdds
from src.fetchers.understat_fetcher import UnderstatFetcher, TeamXGData
from src.fetchers.clubelo_fetcher import ClubELOFetcher, TeamELO
from src.fetchers.polymarket_fetcher import PolymarketFetcher, PolymarketPrices


class TestBetfairOdds:
    """Test BetfairOdds dataclass."""

    def test_betfair_odds_creation(self):
        """Test creating BetfairOdds instance."""
        odds = BetfairOdds(
            match_id="12345",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc),
            home_win_odds=2.10,
            draw_odds=3.50,
            away_win_odds=3.60,
        )

        assert odds.match_id == "12345"
        assert odds.home_team == "Arsenal"
        assert odds.home_win_odds == 2.10
        assert odds.over_2_5_odds is None  # Optional field

    def test_betfair_odds_with_ou_btts(self):
        """Test BetfairOdds with Over/Under and BTTS."""
        odds = BetfairOdds(
            match_id="12346",
            home_team="Liverpool",
            away_team="Man City",
            kickoff=datetime(2026, 1, 20, 17, 30, tzinfo=timezone.utc),
            home_win_odds=3.00,
            draw_odds=3.40,
            away_win_odds=2.30,
            over_2_5_odds=1.80,
            under_2_5_odds=2.10,
            btts_yes_odds=1.75,
            btts_no_odds=2.15,
        )

        assert odds.over_2_5_odds == 1.80
        assert odds.btts_yes_odds == 1.75


class TestBetfairFetcher:
    """Test BetfairFetcher class."""

    def test_betfair_fetcher_init(self):
        """Test fetcher initialization."""
        with patch('src.fetchers.betfair_fetcher.betfairlightweight.APIClient') as mock_client:
            fetcher = BetfairFetcher()
            assert fetcher._logged_in is False
            mock_client.assert_called_once()

    def test_betfair_fetcher_context_manager(self):
        """Test fetcher context manager."""
        with patch('src.fetchers.betfair_fetcher.betfairlightweight.APIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            fetcher = BetfairFetcher()
            with fetcher:
                mock_client.login.assert_called_once()

            mock_client.logout.assert_called_once()


class TestTeamXGData:
    """Test TeamXGData dataclass."""

    def test_team_xg_data_creation(self):
        """Test creating TeamXGData instance."""
        data = TeamXGData(
            team_name="Arsenal",
            matches_played=20,
            xg_for=1.85,
            xga_against=0.95,
            xpts=2.15,
            home_xg=2.10,
            home_xga=0.80,
            away_xg=1.60,
            away_xga=1.10,
            form_xg=[2.1, 1.8, 2.3, 1.5, 1.9],
            form_xga=[0.9, 1.2, 0.7, 1.1, 0.8],
        )

        assert data.team_name == "Arsenal"
        assert data.xg_for == 1.85
        assert len(data.form_xg) == 5


class TestUnderstatFetcher:
    """Test UnderstatFetcher class."""

    def test_understat_fetcher_init(self):
        """Test fetcher initialization."""
        fetcher = UnderstatFetcher(season=2025)
        assert fetcher.season == 2025
        assert fetcher.LEAGUE == "epl"
        assert fetcher.FORM_WINDOW == 5
        assert fetcher.ROLLING_WINDOW == 10

    def test_xg_to_xpts_calculation(self):
        """Test xG to expected points conversion."""
        fetcher = UnderstatFetcher()

        # Home team with higher xG should get more expected points
        home_xpts = fetcher._xg_to_xpts(xg_for=2.0, xg_against=1.0, is_home=True)
        away_xpts = fetcher._xg_to_xpts(xg_for=2.0, xg_against=1.0, is_home=False)

        # Home advantage should give higher xPTS
        assert home_xpts > away_xpts

        # Higher xG advantage should give more points
        high_xg_xpts = fetcher._xg_to_xpts(xg_for=3.0, xg_against=0.5, is_home=True)
        low_xg_xpts = fetcher._xg_to_xpts(xg_for=1.0, xg_against=2.0, is_home=True)
        assert high_xg_xpts > low_xg_xpts


class TestTeamELO:
    """Test TeamELO dataclass."""

    def test_team_elo_creation(self):
        """Test creating TeamELO instance."""
        elo = TeamELO(
            team_name="Manchester City",
            elo_rating=1950.5,
            rank=1,
        )

        assert elo.team_name == "Manchester City"
        assert elo.elo_rating == 1950.5
        assert elo.rank == 1


class TestClubELOFetcher:
    """Test ClubELOFetcher class."""

    def test_clubelo_fetcher_team_mapping(self):
        """Test team name mapping."""
        fetcher = ClubELOFetcher()

        assert fetcher.TEAM_MAPPING["Man City"] == "Manchester City"
        assert fetcher.TEAM_MAPPING["Spurs"] == "Tottenham"
        assert fetcher.TEAM_MAPPING["Wolves"] == "Wolverhampton Wanderers"

    def test_clubelo_win_probability_calculation(self):
        """Test ELO to win probability conversion."""
        fetcher = ClubELOFetcher()

        # Higher rated home team should have higher win probability
        probs = fetcher.calculate_win_probability(home_elo=1900, away_elo=1700)

        assert probs["home"] > probs["away"]
        assert probs["home"] + probs["draw"] + probs["away"] == pytest.approx(1.0, abs=0.001)

        # Equal ELO should be close to 50/50 with slight home advantage
        equal_probs = fetcher.calculate_win_probability(home_elo=1800, away_elo=1800)
        assert equal_probs["home"] > equal_probs["away"]  # Home advantage

    def test_clubelo_draw_probability_bounds(self):
        """Test draw probability stays within bounds."""
        fetcher = ClubELOFetcher()

        # Large ELO difference
        large_diff = fetcher.calculate_win_probability(home_elo=2000, away_elo=1500)
        assert 0.15 <= large_diff["draw"] <= 0.32

        # Small ELO difference
        small_diff = fetcher.calculate_win_probability(home_elo=1800, away_elo=1790)
        assert 0.15 <= small_diff["draw"] <= 0.32

    def test_get_team_elo_direct_match(self):
        """Test getting team ELO with direct name match."""
        fetcher = ClubELOFetcher()
        ratings = {
            "Arsenal": TeamELO("Arsenal", 1920.0, 2),
            "Liverpool": TeamELO("Liverpool", 1940.0, 1),
        }

        result = fetcher.get_team_elo("Arsenal", ratings)
        assert result is not None
        assert result.elo_rating == 1920.0

    def test_get_team_elo_fuzzy_match(self):
        """Test getting team ELO with fuzzy name match."""
        fetcher = ClubELOFetcher()
        ratings = {
            "Manchester City": TeamELO("Manchester City", 1950.0, 1),
            "Manchester United": TeamELO("Manchester United", 1850.0, 5),
        }

        # Should match "City" to "Manchester City"
        result = fetcher.get_team_elo("City", ratings)
        assert result is not None
        assert result.team_name == "Manchester City"


class TestPolymarketPrices:
    """Test PolymarketPrices dataclass."""

    def test_polymarket_prices_creation(self):
        """Test creating PolymarketPrices instance."""
        prices = PolymarketPrices(
            match_id="pm-12345",
            home_team="Arsenal",
            away_team="Chelsea",
            home_win_price=0.55,
            draw_price=0.25,
            away_win_price=0.20,
        )

        assert prices.match_id == "pm-12345"
        assert prices.home_win_price == 0.55
        assert prices.over_2_5_price is None  # Optional

    def test_polymarket_prices_with_liquidity(self):
        """Test PolymarketPrices with liquidity data."""
        prices = PolymarketPrices(
            match_id="pm-12346",
            home_team="Liverpool",
            away_team="Man City",
            home_win_price=0.35,
            draw_price=0.30,
            away_win_price=0.35,
            home_win_liquidity=10000.0,
            over_2_5_liquidity=5000.0,
        )

        assert prices.home_win_liquidity == 10000.0
        assert prices.over_2_5_liquidity == 5000.0


class TestPolymarketFetcher:
    """Test PolymarketFetcher class."""

    def test_polymarket_fetcher_parse_teams_vs_pattern(self):
        """Test parsing teams from 'vs' pattern."""
        fetcher = PolymarketFetcher()

        home, away = fetcher._parse_teams("Arsenal vs Chelsea")
        assert home == "Arsenal"
        assert away == "Chelsea"

        home, away = fetcher._parse_teams("Liverpool v Man City - Winner")
        assert home == "Liverpool"
        assert away == "Man City"

    def test_polymarket_fetcher_parse_teams_win_pattern(self):
        """Test parsing teams from 'Will X win' pattern."""
        fetcher = PolymarketFetcher()

        home, away = fetcher._parse_teams("Will Arsenal win against Chelsea?")
        assert home == "Arsenal"
        assert away == "Chelsea"

        home, away = fetcher._parse_teams("Will Liverpool beat Everton?")
        assert home == "Liverpool"
        assert away == "Everton"

    def test_polymarket_fetcher_parse_teams_no_match(self):
        """Test parsing teams returns empty on no match."""
        fetcher = PolymarketFetcher()

        home, away = fetcher._parse_teams("Random question about football")
        assert home == ""
        assert away == ""

    def test_polymarket_fetcher_epl_teams_list(self):
        """Test EPL teams list is complete."""
        fetcher = PolymarketFetcher()

        assert len(fetcher.EPL_TEAMS) == 20
        assert "Arsenal" in fetcher.EPL_TEAMS
        assert "Liverpool" in fetcher.EPL_TEAMS
        assert "Manchester City" in fetcher.EPL_TEAMS
