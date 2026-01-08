"""Configuration module initialization."""
import logging
import os
import sys
from pathlib import Path


def load_streamlit_secrets():
    """Load Streamlit Cloud secrets into environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # Betfair
            if "betfair" in st.secrets:
                bf = st.secrets["betfair"]
                if bf.get("username"):
                    os.environ.setdefault("BETFAIR_USERNAME", bf["username"])
                if bf.get("password"):
                    os.environ.setdefault("BETFAIR_PASSWORD", bf["password"])
                if bf.get("app_key"):
                    os.environ.setdefault("BETFAIR_APP_KEY", bf["app_key"])

            # Telegram
            if "telegram" in st.secrets:
                tg = st.secrets["telegram"]
                if tg.get("bot_token"):
                    os.environ.setdefault("TELEGRAM_BOT_TOKEN", tg["bot_token"])
                if tg.get("chat_id"):
                    os.environ.setdefault("TELEGRAM_CHAT_ID", tg["chat_id"])

            # Polymarket
            if "polymarket" in st.secrets:
                pm = st.secrets["polymarket"]
                if pm.get("private_key"):
                    os.environ.setdefault("POLYMARKET_PRIVATE_KEY", pm["private_key"])

            # Football Data
            if "football_data" in st.secrets:
                fd = st.secrets["football_data"]
                if fd.get("api_key"):
                    os.environ.setdefault("FOOTBALL_DATA_API_KEY", fd["api_key"])

            # DomeAPI (Polymarket wrapper)
            if "dome_api" in st.secrets:
                dome = st.secrets["dome_api"]
                if dome.get("api_key"):
                    os.environ.setdefault("DOME_API_KEY", dome["api_key"])
            # Also support flat key format: DOME_API_KEY = "xxx"
            elif "DOME_API_KEY" in st.secrets:
                os.environ.setdefault("DOME_API_KEY", st.secrets["DOME_API_KEY"])
    except Exception:
        pass  # Not running in Streamlit or no secrets configured


# Load secrets before importing settings
load_streamlit_secrets()

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
