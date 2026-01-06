"""Tests for configuration module."""
import pytest
from pathlib import Path


class TestSettings:
    """Test settings loading and validation."""

    def test_settings_loads(self):
        """Ensure settings module loads without error."""
        from src.config.settings import settings
        assert settings is not None

    def test_default_values(self):
        """Test default settings values."""
        from src.config.settings import settings

        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.scan_interval_minutes == 15
        assert settings.edge_threshold_1x2 == 0.05
        assert settings.edge_threshold_ou == 0.07
        assert settings.edge_threshold_btts == 0.07

    def test_database_path_is_path(self):
        """Test database_path is a Path object."""
        from src.config.settings import settings
        assert isinstance(settings.database_path, Path)


class TestWeightConfig:
    """Test weight configuration loading."""

    def test_load_1x2_weights(self):
        """Test loading 1X2 weight configuration."""
        from src.config.settings import load_weight_config

        config = load_weight_config("1x2")

        assert "market_trust" in config
        assert "balanced" in config
        assert "analytics_first" in config
        assert "time_thresholds" in config

    def test_load_ou_weights(self):
        """Test loading Over/Under weight configuration."""
        from src.config.settings import load_weight_config

        config = load_weight_config("ou")

        assert "market_trust" in config
        assert config["market_trust"]["xg"] > 0

    def test_load_btts_weights(self):
        """Test loading BTTS weight configuration."""
        from src.config.settings import load_weight_config

        config = load_weight_config("btts")

        assert "market_trust" in config
        # BTTS uses xga instead of elo
        assert "xga" in config["market_trust"]

    def test_weights_sum_to_one(self):
        """Test that weights in each profile sum to 1.0."""
        from src.config.settings import load_weight_config

        for market in ["1x2", "ou", "btts"]:
            config = load_weight_config(market)
            for profile_name, profile in config.items():
                if profile_name == "time_thresholds":
                    continue
                weights = {k: v for k, v in profile.items() if k != "description"}
                total = sum(weights.values())
                assert abs(total - 1.0) < 0.001, f"{market}/{profile_name} sums to {total}"

    def test_invalid_market_raises_error(self):
        """Test that invalid market type raises FileNotFoundError."""
        from src.config.settings import load_weight_config

        with pytest.raises(FileNotFoundError):
            load_weight_config("invalid_market")


class TestGetWeightProfile:
    """Test time-based weight profile selection."""

    def test_analytics_first_profile(self):
        """Test selection of analytics_first profile (>96 hours)."""
        from src.config.settings import get_weight_profile

        weights = get_weight_profile("1x2", hours_to_kickoff=120)

        # Analytics first: betfair=0.30, xg=0.35, elo=0.25, form=0.10
        assert weights["betfair"] == 0.30
        assert weights["xg"] == 0.35

    def test_balanced_profile(self):
        """Test selection of balanced profile (24-96 hours)."""
        from src.config.settings import get_weight_profile

        weights = get_weight_profile("1x2", hours_to_kickoff=48)

        # Balanced: betfair=0.40, xg=0.30, elo=0.20, form=0.10
        assert weights["betfair"] == 0.40
        assert weights["xg"] == 0.30

    def test_market_trust_profile(self):
        """Test selection of market_trust profile (<24 hours)."""
        from src.config.settings import get_weight_profile

        weights = get_weight_profile("1x2", hours_to_kickoff=12)

        # Market trust: betfair=0.55, xg=0.20, elo=0.15, form=0.10
        assert weights["betfair"] == 0.55
        assert weights["xg"] == 0.20


class TestLogging:
    """Test logging configuration."""

    def test_setup_logging(self):
        """Test logging setup creates handlers."""
        import logging
        from src.config import setup_logging

        setup_logging()
        logger = logging.getLogger()

        # Should have at least stdout and file handlers
        assert len(logger.handlers) >= 1
