"""Main entry point for EPL Bet Indicator."""
import asyncio
import logging
import signal
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Global event loop reference for signal handlers
_loop: Optional[asyncio.AbstractEventLoop] = None
_shutdown_event: Optional[asyncio.Event] = None


async def startup():
    """Startup routine."""
    from src.config import setup_logging
    from src.alerts import TelegramAlertService
    from src.services import get_scheduler
    from src.storage.database import db

    setup_logging()
    logger.info("Starting EPL Bet Indicator")

    # Initialize database
    db.create_tables()

    # Send startup notification
    telegram = TelegramAlertService()
    if telegram.is_configured:
        await telegram.send_startup_alert()
        logger.info("Startup alert sent")
    else:
        logger.warning("Telegram not configured, skipping startup alert")

    # Start scheduler
    scheduler = get_scheduler()
    scheduler.start()

    # Run initial scan
    logger.info("Running initial scan...")
    await scheduler.scan_and_alert()

    logger.info("Startup complete")


async def shutdown():
    """Shutdown routine."""
    from src.services import get_scheduler
    from src.alerts import TelegramAlertService
    from src.alerts.templates import format_template
    from datetime import datetime

    logger.info("Shutting down...")

    # Stop scheduler
    scheduler = get_scheduler()
    scheduler.stop()

    # Send shutdown notification (optional)
    telegram = TelegramAlertService()
    if telegram.is_configured:
        message = format_template(
            "shutdown",
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        try:
            await telegram.send_alert(message)
        except Exception:
            pass  # Don't fail shutdown due to telegram

    logger.info("Shutdown complete")


def _signal_handler(signum, frame):
    """Handle termination signals."""
    global _shutdown_event
    logger.info(f"Received signal {signum}")
    if _shutdown_event:
        _shutdown_event.set()


async def run_service():
    """Run the main service loop."""
    global _shutdown_event

    _shutdown_event = asyncio.Event()

    # Setup signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _signal_handler)

    try:
        # Startup
        await startup()

        # Wait for shutdown signal
        await _shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("Service cancelled")
    finally:
        # Shutdown
        await shutdown()


def main():
    """Main function - entry point for service."""
    global _loop

    try:
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(run_service())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        if _loop:
            _loop.close()


if __name__ == "__main__":
    main()
