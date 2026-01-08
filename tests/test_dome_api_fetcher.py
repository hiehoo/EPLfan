"""Tests for DomeAPI fetcher."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.fetchers.dome_api_fetcher import DomeApiFetcher, DomeApiPrices


class TestDomeApiPrices:
    """Test DomeApiPrices dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating DomeApiPrices with required fields."""
        prices = DomeApiPrices(
            market_slug="arsenal-vs-chelsea",
            condition_id="0x123",
            home_team="Arsenal",
            away_team="Chelsea",
            home_win_price=0.55,
            draw_price=0.25,
            away_win_price=0.20,
        )

        assert prices.home_team == "Arsenal"
        assert prices.away_team == "Chelsea"
        assert prices.home_win_price == 0.55
        assert prices.draw_price == 0.25
        assert prices.away_win_price == 0.20
        assert prices.over_2_5_price is None
        assert prices.btts_yes_price is None

    def test_creation_with_all_fields(self):
        """Test creating DomeApiPrices with all market types."""
        prices = DomeApiPrices(
            market_slug="arsenal-vs-chelsea",
            condition_id="0x123",
            home_team="Arsenal",
            away_team="Chelsea",
            home_win_price=0.55,
            draw_price=0.25,
            away_win_price=0.20,
            over_2_5_price=0.60,
            under_2_5_price=0.40,
            btts_yes_price=0.55,
            btts_no_price=0.45,
            volume=100000,
            liquidity=50000,
        )

        assert prices.over_2_5_price == 0.60
        assert prices.btts_yes_price == 0.55
        assert prices.volume == 100000
        assert prices.liquidity == 50000


class TestDomeApiFetcher:
    """Test DomeApiFetcher class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with configured API key."""
        with patch("src.fetchers.dome_api_fetcher.settings") as mock:
            mock.dome_api.is_configured.return_value = True
            mock.dome_api.api_key.get_secret_value.return_value = "test_api_key"
            yield mock

    @pytest.fixture
    def fetcher(self, mock_settings):
        """Create fetcher with mocked settings."""
        return DomeApiFetcher()

    def test_parse_teams_vs_pattern(self, fetcher):
        """Test team parsing with 'vs' pattern."""
        home, away = fetcher._parse_teams("Arsenal vs Chelsea")
        assert home == "Arsenal"
        assert away == "Chelsea"

    def test_parse_teams_v_pattern(self, fetcher):
        """Test team parsing with 'v' pattern."""
        home, away = fetcher._parse_teams("Liverpool v Man City - Winner")
        assert home == "Liverpool"
        assert away == "Man City"

    def test_parse_teams_will_win_pattern(self, fetcher):
        """Test team parsing with 'Will X win' pattern."""
        home, away = fetcher._parse_teams("Will Arsenal win against Chelsea?")
        assert home == "Arsenal"
        assert away == "Chelsea"

    def test_parse_teams_no_match(self, fetcher):
        """Test team parsing with no matching pattern."""
        home, away = fetcher._parse_teams("Premier League champion 2025")
        assert home == ""
        assert away == ""

    def test_extract_prices_valid_market(self, fetcher):
        """Test price extraction from valid market data."""
        market = {
            "question": "Arsenal vs Chelsea",
            "market_slug": "arsenal-vs-chelsea",
            "condition_id": "0x123",
            "tokens": [
                {"outcome": "Arsenal", "price": 0.55},
                {"outcome": "Draw", "price": 0.25},
                {"outcome": "Chelsea", "price": 0.20},
            ],
            "volume": 100000,
            "liquidity": 50000,
        }

        prices = fetcher._extract_prices(market)

        assert prices is not None
        assert prices.home_team == "Arsenal"
        assert prices.away_team == "Chelsea"
        assert prices.home_win_price == 0.55
        assert prices.draw_price == 0.25
        assert prices.away_win_price == 0.20
        assert prices.volume == 100000

    def test_extract_prices_invalid_market(self, fetcher):
        """Test price extraction from invalid market data."""
        market = {
            "question": "Premier League champion 2025",
            "tokens": [],
        }

        prices = fetcher._extract_prices(market)
        assert prices is None

    @pytest.mark.asyncio
    async def test_fetch_epl_markets_success(self, fetcher):
        """Test successful market fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "markets": [
                {
                    "question": "Arsenal vs Chelsea",
                    "market_slug": "arsenal-vs-chelsea",
                    "condition_id": "0x123",
                    "tokens": [
                        {"outcome": "Arsenal", "price": 0.55},
                        {"outcome": "Draw", "price": 0.25},
                        {"outcome": "Chelsea", "price": 0.20},
                    ],
                    "volume": 100000,
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(fetcher.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            results = await fetcher.fetch_epl_markets()

        assert len(results) == 1
        assert results[0].home_team == "Arsenal"
        assert results[0].away_team == "Chelsea"
        assert results[0].home_win_price == 0.55

    @pytest.mark.asyncio
    async def test_fetch_epl_markets_not_configured(self):
        """Test fetch when API not configured."""
        with patch("src.fetchers.dome_api_fetcher.settings") as mock:
            mock.dome_api.is_configured.return_value = False
            mock.dome_api.api_key.get_secret_value.return_value = ""

            fetcher = DomeApiFetcher()
            results = await fetcher.fetch_epl_markets()

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_epl_markets_api_error(self, fetcher):
        """Test handling of API errors."""
        with patch.object(fetcher.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("API Error")
            results = await fetcher.fetch_epl_markets()

        assert results == []

    @pytest.mark.asyncio
    async def test_close(self, fetcher):
        """Test client cleanup."""
        with patch.object(fetcher.client, "aclose", new_callable=AsyncMock) as mock_close:
            await fetcher.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test async context manager usage."""
        with patch("src.fetchers.dome_api_fetcher.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with DomeApiFetcher() as fetcher:
                assert fetcher is not None

            mock_client.aclose.assert_called_once()


class TestDomeApiFetcherIntegration:
    """Integration tests (require real API key)."""

    @pytest.mark.skip(reason="Requires real API key")
    @pytest.mark.asyncio
    async def test_real_api_fetch(self):
        """Test with real DomeAPI (manual run only)."""
        async with DomeApiFetcher() as fetcher:
            results = await fetcher.fetch_epl_markets()
            print(f"Found {len(results)} EPL markets")
            for r in results[:3]:
                print(f"  {r.home_team} vs {r.away_team}: H={r.home_win_price}")
