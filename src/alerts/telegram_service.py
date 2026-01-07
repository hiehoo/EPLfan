"""Telegram alert service for EPL Bet Indicator."""
import asyncio
import logging
from typing import Optional

from src.config.settings import settings
from src.alerts.templates import (
    SignalAlert,
    format_signal,
    format_batch_signals,
    format_template,
)

logger = logging.getLogger(__name__)


class TelegramAlertService:
    """Telegram alert service for sending betting signals."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """Initialize Telegram service.

        Args:
            bot_token: Override bot token (for testing)
            chat_id: Override chat ID (for testing)
        """
        self._bot_token = bot_token or settings.telegram.bot_token.get_secret_value()
        self._chat_id = chat_id or settings.telegram.chat_id
        self._bot = None

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self._bot_token and self._chat_id)

    async def _get_bot(self):
        """Get or create bot instance (lazy initialization)."""
        if self._bot is None:
            try:
                from telegram import Bot

                self._bot = Bot(token=self._bot_token)
            except ImportError:
                logger.error("python-telegram-bot not installed")
                raise
        return self._bot

    async def send_alert(self, message: str) -> bool:
        """Send alert message to configured chat.

        Args:
            message: HTML-formatted message to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured:
            logger.warning("Telegram not configured, skipping alert")
            return False

        try:
            bot = await self._get_bot()
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode="HTML",
            )
            logger.info("Alert sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    async def send_signal_alert(self, signal: SignalAlert) -> bool:
        """Send formatted signal alert.

        Args:
            signal: SignalAlert object

        Returns:
            True if sent successfully
        """
        message = format_signal(signal)
        return await self.send_alert(message)

    async def send_batch_alerts(self, signals: list[SignalAlert]) -> bool:
        """Send batch of signals as single message.

        Args:
            signals: List of SignalAlert objects

        Returns:
            True if sent successfully
        """
        if not signals:
            logger.info("No signals to send")
            return True

        message = format_batch_signals(signals)
        return await self.send_alert(message)

    async def send_startup_alert(self) -> bool:
        """Send startup notification."""
        message = format_template(
            "startup",
            scan_interval=settings.scan_interval_minutes,
            threshold_1x2=settings.edge_threshold_1x2 * 100,
            threshold_ou=settings.edge_threshold_ou * 100,
            threshold_btts=settings.edge_threshold_btts * 100,
        )
        return await self.send_alert(message)

    async def send_error_alert(self, error_type: str, error_message: str) -> bool:
        """Send error notification.

        Args:
            error_type: Type of error (e.g., "FetchError", "AnalysisError")
            error_message: Error details

        Returns:
            True if sent successfully
        """
        from datetime import datetime

        message = format_template(
            "error",
            error_type=error_type,
            error_message=error_message[:200],  # Truncate long messages
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        return await self.send_alert(message)

    async def send_no_signals_alert(self, next_scan_minutes: int) -> bool:
        """Send 'no signals' notification.

        Args:
            next_scan_minutes: Minutes until next scan

        Returns:
            True if sent successfully
        """
        message = format_template(
            "no_signals",
            next_scan_minutes=next_scan_minutes,
        )
        return await self.send_alert(message)

    async def close(self):
        """Clean up bot resources."""
        if self._bot:
            # Bot doesn't need explicit cleanup in python-telegram-bot v20+
            self._bot = None


# Convenience function for synchronous contexts
def send_alert_sync(message: str) -> bool:
    """Send alert synchronously (convenience wrapper).

    Args:
        message: Message to send

    Returns:
        True if sent successfully
    """
    service = TelegramAlertService()
    return asyncio.run(service.send_alert(message))
