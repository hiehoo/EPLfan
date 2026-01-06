"""Configuration module initialization."""
import logging
import sys
from pathlib import Path

from .settings import settings, load_weight_config, get_weight_profile


def setup_logging() -> None:
    """Configure application logging."""
    # Ensure data directory exists for log file
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log", mode="a"),
        ],
    )


# Auto-setup on import
setup_logging()

__all__ = ["settings", "setup_logging", "load_weight_config", "get_weight_profile"]
