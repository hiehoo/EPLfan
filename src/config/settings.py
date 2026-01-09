"""Application settings with validation."""
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class BetfairSettings(BaseSettings):
    """Betfair API configuration."""

    model_config = SettingsConfigDict(env_prefix="BETFAIR_")

    username: str = ""
    password: SecretStr = SecretStr("")
    app_key: SecretStr = SecretStr("")
    certs_path: Path = Path("./certs")

    def is_configured(self) -> bool:
        """Check if Betfair credentials are configured."""
        return bool(
            self.username
            and self.password.get_secret_value()
            and self.app_key.get_secret_value()
        )


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")

    bot_token: SecretStr = SecretStr("")
    chat_id: str = ""

    def is_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.bot_token.get_secret_value() and self.chat_id)


class PolymarketSettings(BaseSettings):
    """Polymarket configuration."""

    model_config = SettingsConfigDict(env_prefix="POLYMARKET_")

    private_key: SecretStr = SecretStr("")

    def is_configured(self) -> bool:
        """Check if Polymarket private key is configured."""
        return bool(self.private_key.get_secret_value())


class DomeApiSettings(BaseSettings):
    """DomeAPI configuration (Polymarket wrapper)."""

    model_config = SettingsConfigDict(
        env_prefix="DOME_",
        env_file=".env",
        extra="ignore"
    )

    api_key: SecretStr = SecretStr("")

    def is_configured(self) -> bool:
        """Check if DomeAPI key is configured."""
        return bool(self.api_key.get_secret_value())


class FootballDataSettings(BaseSettings):
    """Football-Data.org API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="FOOTBALL_DATA_",
        env_file=".env",
        extra="ignore"
    )

    api_key: str = ""

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)


class WindowConfig(BaseSettings):
    """Rolling window configuration."""
    model_config = SettingsConfigDict(env_prefix="WINDOW_")

    window_1x2: int = 10
    window_ou_2_5: int = 6
    window_btts: int = 4
    window_shots: int = 5
    window_xg: int = 8


class AppSettings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"
    database_path: Path = Path("./data/epl_indicator.db")
    scan_interval_minutes: int = 15

    # Edge thresholds
    edge_threshold_1x2: float = 0.05  # 5% (profitable threshold based on backtest)
    edge_threshold_ou: float = 0.07   # 7%
    edge_threshold_btts: float = 0.07  # 7%

    # Nested settings
    betfair: BetfairSettings = Field(default_factory=BetfairSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    dome_api: DomeApiSettings = Field(default_factory=DomeApiSettings)
    football_data: FootballDataSettings = Field(default_factory=FootballDataSettings)
    windows: WindowConfig = Field(default_factory=WindowConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


def load_weight_config(market_type: str) -> dict:
    """Load weight configuration for a market type.

    Args:
        market_type: One of '1x2', 'ou', 'btts'

    Returns:
        Dictionary with weight profiles and time thresholds
    """
    weights_dir = Path(__file__).parent / "weights"
    weight_file = weights_dir / f"{market_type}_weights.yaml"

    if not weight_file.exists():
        raise FileNotFoundError(f"Weight config not found: {weight_file}")

    try:
        with open(weight_file) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load weight config {weight_file}: {e}")

    # Validate weights sum to 1.0 for each profile
    for profile_name, profile in config.items():
        if profile_name == "time_thresholds":
            continue
        weights = {k: v for k, v in profile.items() if k != "description"}
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Weights in {market_type}/{profile_name} sum to {total}, expected 1.0"
            )

    return config


def get_weight_profile(market_type: str, hours_to_kickoff: float) -> dict:
    """Get appropriate weight profile based on time to kickoff.

    Args:
        market_type: One of '1x2', 'ou', 'btts'
        hours_to_kickoff: Hours until match starts

    Returns:
        Dictionary with weights for each data source
    """
    config = load_weight_config(market_type)
    thresholds = config.get("time_thresholds", {})

    # Determine profile based on time
    if hours_to_kickoff > thresholds.get("analytics_first", 96):
        profile_name = "analytics_first"
    elif hours_to_kickoff > thresholds.get("balanced", 24):
        profile_name = "balanced"
    else:
        profile_name = "market_trust"

    profile = config[profile_name]
    return {k: v for k, v in profile.items() if k != "description"}


def load_window_config() -> dict:
    """Load window configuration from YAML."""
    windows_file = Path(__file__).parent / "weights" / "windows.yaml"
    if windows_file.exists():
        with open(windows_file) as f:
            return yaml.safe_load(f)
    return {}


# Singleton instance
settings = AppSettings()
