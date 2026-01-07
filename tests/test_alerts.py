"""Tests for alert system."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.alerts.templates import (
    SignalAlert,
    format_template,
    format_signal,
    format_batch_signals,
    get_edge_emoji,
    get_market_emoji,
    TEMPLATES,
)
from src.alerts.telegram_service import TelegramAlertService


class TestSignalAlert:
    """Tests for SignalAlert dataclass."""

    def test_signal_alert_creation(self):
        """Test SignalAlert creation."""
        alert = SignalAlert(
            match_id=1,
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff="2026-01-10 15:00 UTC",
            market="1X2",
            selection="HOME",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            weight_profile="market_trust",
        )

        assert alert.match_id == 1
        assert alert.home_team == "Arsenal"
        assert alert.edge == 0.07
        assert alert.confidence == "medium"  # default

    def test_signal_alert_to_dict(self):
        """Test SignalAlert to_dict method."""
        alert = SignalAlert(
            match_id=1,
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff="2026-01-10 15:00 UTC",
            market="1X2",
            selection="HOME",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            weight_profile="market_trust",
        )

        d = alert.to_dict()
        assert d["match_id"] == 1
        assert d["home_team"] == "Arsenal"
        assert d["edge"] == 0.07


class TestTemplates:
    """Tests for alert templates."""

    def test_templates_exist(self):
        """Test all expected templates exist."""
        expected = [
            "signal_single",
            "signal_batch_header",
            "signal_batch_line",
            "daily_summary",
            "error",
            "startup",
            "shutdown",
            "no_signals",
        ]
        for name in expected:
            assert name in TEMPLATES, f"Missing template: {name}"

    def test_get_edge_emoji_strong(self):
        """Test strong edge emoji."""
        assert get_edge_emoji(0.12) == "🟢"
        assert get_edge_emoji(0.10) == "🟢"

    def test_get_edge_emoji_moderate(self):
        """Test moderate edge emoji."""
        assert get_edge_emoji(0.08) == "🟡"
        assert get_edge_emoji(0.07) == "🟡"

    def test_get_edge_emoji_marginal(self):
        """Test marginal edge emoji."""
        assert get_edge_emoji(0.05) == "🟠"
        assert get_edge_emoji(0.03) == "🟠"

    def test_get_market_emoji_1x2(self):
        """Test 1X2 market emoji."""
        assert get_market_emoji("1X2") == "⚽"
        assert get_market_emoji("1x2") == "⚽"

    def test_get_market_emoji_ou(self):
        """Test O/U market emoji."""
        assert get_market_emoji("O/U 2.5") == "📈"
        assert get_market_emoji("over") == "📈"
        assert get_market_emoji("under") == "📈"

    def test_get_market_emoji_btts(self):
        """Test BTTS market emoji."""
        assert get_market_emoji("BTTS") == "🎯"
        assert get_market_emoji("btts") == "🎯"

    def test_get_market_emoji_default(self):
        """Test default market emoji."""
        assert get_market_emoji("unknown") == "📊"


class TestFormatTemplate:
    """Tests for format_template function."""

    def test_format_startup_template(self):
        """Test startup template formatting."""
        result = format_template(
            "startup",
            scan_interval=15,
            threshold_1x2=5.0,
            threshold_ou=7.0,
            threshold_btts=7.0,
        )

        assert "EPL Bet Indicator Started" in result
        assert "15 min" in result
        assert "5%" in result
        assert "7%" in result

    def test_format_error_template(self):
        """Test error template formatting."""
        result = format_template(
            "error",
            error_type="FetchError",
            error_message="Connection timeout",
            timestamp="2026-01-10 15:00 UTC",
        )

        assert "System Alert" in result
        assert "FetchError" in result
        assert "Connection timeout" in result

    def test_format_unknown_template(self):
        """Test unknown template returns error."""
        result = format_template("nonexistent")
        assert "Unknown template" in result

    def test_format_missing_key(self):
        """Test missing key returns error."""
        result = format_template("startup", scan_interval=15)
        assert "Missing template key" in result


class TestFormatSignal:
    """Tests for format_signal function."""

    def test_format_signal_basic(self):
        """Test basic signal formatting."""
        alert = SignalAlert(
            match_id=1,
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff="2026-01-10 15:00 UTC",
            market="1X2",
            selection="HOME",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            weight_profile="market_trust",
        )

        result = format_signal(alert)

        assert "Arsenal vs Chelsea" in result
        assert "1X2" in result
        assert "HOME" in result
        assert "55.0%" in result
        assert "48.0%" in result
        assert "7.0%" in result
        assert "market_trust" in result

    def test_format_signal_contains_html_tags(self):
        """Test signal contains HTML formatting."""
        alert = SignalAlert(
            match_id=1,
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff="2026-01-10 15:00 UTC",
            market="1X2",
            selection="HOME",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.10,
            weight_profile="market_trust",
        )

        result = format_signal(alert)

        assert "<b>" in result
        assert "<code>" in result
        assert "<i>" in result


class TestFormatBatchSignals:
    """Tests for format_batch_signals function."""

    def test_format_batch_signals_empty(self):
        """Test empty signals list."""
        result = format_batch_signals([])
        assert "No signals" in result

    def test_format_batch_signals_single(self):
        """Test single signal in batch."""
        alerts = [
            SignalAlert(
                match_id=1,
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff="2026-01-10 15:00 UTC",
                market="1X2",
                selection="HOME",
                fair_prob=0.55,
                market_prob=0.48,
                edge=0.07,
                weight_profile="market_trust",
            )
        ]

        result = format_batch_signals(alerts)

        assert "1 signal(s) found" in result
        assert "ARS v CHE" in result
        assert "7.0%" in result

    def test_format_batch_signals_multiple(self):
        """Test multiple signals in batch."""
        alerts = [
            SignalAlert(
                match_id=1,
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff="2026-01-10 15:00 UTC",
                market="1X2",
                selection="HOME",
                fair_prob=0.55,
                market_prob=0.48,
                edge=0.07,
                weight_profile="market_trust",
            ),
            SignalAlert(
                match_id=2,
                home_team="Liverpool",
                away_team="Manchester City",
                kickoff="2026-01-10 17:30 UTC",
                market="O/U 2.5",
                selection="OVER",
                fair_prob=0.60,
                market_prob=0.52,
                edge=0.08,
                weight_profile="balanced",
            ),
        ]

        result = format_batch_signals(alerts)

        assert "2 signal(s) found" in result
        assert "ARS v CHE" in result
        assert "LIV v MAN" in result

    def test_format_batch_signals_sorted_by_edge(self):
        """Test signals are sorted by edge descending."""
        alerts = [
            SignalAlert(
                match_id=1,
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff="2026-01-10 15:00 UTC",
                market="1X2",
                selection="HOME",
                fair_prob=0.55,
                market_prob=0.48,
                edge=0.05,
                weight_profile="market_trust",
            ),
            SignalAlert(
                match_id=2,
                home_team="Liverpool",
                away_team="Manchester City",
                kickoff="2026-01-10 17:30 UTC",
                market="O/U 2.5",
                selection="OVER",
                fair_prob=0.60,
                market_prob=0.52,
                edge=0.10,
                weight_profile="balanced",
            ),
        ]

        result = format_batch_signals(alerts)

        # Liverpool (10% edge) should appear before Arsenal (5% edge)
        liv_pos = result.find("LIV")
        ars_pos = result.find("ARS")
        assert liv_pos < ars_pos


class TestTelegramAlertService:
    """Tests for TelegramAlertService."""

    def test_service_initialization_with_credentials(self):
        """Test service initializes with provided credentials."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        assert service.is_configured is True

    def test_service_initialization_without_credentials(self):
        """Test service with missing credentials."""
        service = TelegramAlertService(
            bot_token="",
            chat_id="",
        )

        assert service.is_configured is False

    @pytest.mark.asyncio
    async def test_send_alert_not_configured(self):
        """Test send_alert returns False when not configured."""
        service = TelegramAlertService(
            bot_token="",
            chat_id="",
        )

        result = await service.send_alert("Test message")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_success(self):
        """Test send_alert success with mocked bot."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        # Mock the bot
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        with patch.object(service, "_get_bot", return_value=mock_bot):
            result = await service.send_alert("Test message")

        assert result is True
        mock_bot.send_message.assert_called_once_with(
            chat_id="test_chat",
            text="Test message",
            parse_mode="HTML",
        )

    @pytest.mark.asyncio
    async def test_send_signal_alert(self):
        """Test send_signal_alert."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        alert = SignalAlert(
            match_id=1,
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff="2026-01-10 15:00 UTC",
            market="1X2",
            selection="HOME",
            fair_prob=0.55,
            market_prob=0.48,
            edge=0.07,
            weight_profile="market_trust",
        )

        with patch.object(service, "_get_bot", return_value=mock_bot):
            result = await service.send_signal_alert(alert)

        assert result is True
        # Check message contains expected content
        call_args = mock_bot.send_message.call_args
        message = call_args.kwargs["text"]
        assert "Arsenal vs Chelsea" in message

    @pytest.mark.asyncio
    async def test_send_batch_alerts_empty(self):
        """Test send_batch_alerts with empty list."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        result = await service.send_batch_alerts([])
        assert result is True  # Empty list is success

    @pytest.mark.asyncio
    async def test_send_error_alert(self):
        """Test send_error_alert."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        with patch.object(service, "_get_bot", return_value=mock_bot):
            result = await service.send_error_alert(
                "FetchError",
                "Connection timeout"
            )

        assert result is True
        call_args = mock_bot.send_message.call_args
        message = call_args.kwargs["text"]
        assert "FetchError" in message
        assert "Connection timeout" in message

    @pytest.mark.asyncio
    async def test_send_error_alert_truncates_long_message(self):
        """Test error messages are truncated."""
        service = TelegramAlertService(
            bot_token="test_token",
            chat_id="test_chat",
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        long_message = "x" * 500

        with patch.object(service, "_get_bot", return_value=mock_bot):
            await service.send_error_alert("Error", long_message)

        call_args = mock_bot.send_message.call_args
        message = call_args.kwargs["text"]
        # Message should be truncated
        assert len(message) < len(long_message) + 100
