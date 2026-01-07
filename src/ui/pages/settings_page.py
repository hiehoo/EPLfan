"""Settings configuration page."""
import streamlit as st
import yaml
from pathlib import Path

from src.config.settings import settings


def render():
    """Render settings page."""
    st.header("Settings")

    tab1, tab2, tab3 = st.tabs(["General", "Weights", "Alerts"])

    with tab1:
        _render_general_settings()

    with tab2:
        _render_weight_settings()

    with tab3:
        _render_alert_settings()


def _render_general_settings():
    """Render general settings."""
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Scan Interval (minutes)",
            value=settings.scan_interval_minutes,
            min_value=5,
            max_value=60,
            key="scan_interval",
            disabled=True,
            help="Edit in .env file",
        )

    with col2:
        st.selectbox(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings.log_level),
            key="log_level",
            disabled=True,
            help="Edit in .env file",
        )

    st.subheader("Edge Thresholds")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.slider(
            "1X2 Threshold %",
            0.0,
            15.0,
            settings.edge_threshold_1x2 * 100,
            0.5,
            key="threshold_1x2",
            disabled=True,
            help="Edit in .env file",
        )

    with col2:
        st.slider(
            "O/U Threshold %",
            0.0,
            15.0,
            settings.edge_threshold_ou * 100,
            0.5,
            key="threshold_ou",
            disabled=True,
            help="Edit in .env file",
        )

    with col3:
        st.slider(
            "BTTS Threshold %",
            0.0,
            15.0,
            settings.edge_threshold_btts * 100,
            0.5,
            key="threshold_btts",
            disabled=True,
            help="Edit in .env file",
        )

    st.info("Settings are read from environment variables. Edit the .env file and restart to change.")


def _render_weight_settings():
    """Render weight configuration."""
    st.subheader("Weight Profiles")

    market = st.selectbox("Market Type", ["1X2", "Over/Under", "BTTS"])

    weight_file = {
        "1X2": "1x2_weights.yaml",
        "Over/Under": "ou_weights.yaml",
        "BTTS": "btts_weights.yaml",
    }[market]

    weight_path = Path("src/config/weights") / weight_file

    if weight_path.exists():
        with open(weight_path) as f:
            weights = yaml.safe_load(f)

        for profile_name, profile in weights.items():
            if profile_name == "time_thresholds":
                continue

            if not isinstance(profile, dict):
                continue

            with st.expander(f"{profile_name.replace('_', ' ').title()}"):
                st.write(profile.get("description", ""))

                cols = st.columns(4)
                factors = ["betfair", "xg", "elo", "form"]
                if market == "BTTS":
                    factors = ["betfair", "xg", "xga", "form"]

                for idx, factor in enumerate(factors):
                    if factor in profile:
                        with cols[idx]:
                            st.metric(factor.upper(), f"{profile[factor] * 100:.0f}%")

        # Time thresholds
        st.subheader("Time-Based Auto-Selection")
        thresholds = weights.get("time_thresholds", {})
        st.write(f"- **Analytics First:** > {thresholds.get('analytics_first', 96)} hours to kickoff")
        st.write(f"- **Balanced:** > {thresholds.get('balanced', 24)} hours to kickoff")
        st.write(f"- **Market Trust:** <= {thresholds.get('balanced', 24)} hours to kickoff")
    else:
        st.warning(f"Weight file not found: {weight_path}")


def _render_alert_settings():
    """Render alert settings."""
    st.subheader("Telegram Alerts")

    # Show masked credentials
    has_token = bool(settings.telegram.bot_token.get_secret_value()) if settings.telegram.bot_token else False

    st.text_input(
        "Bot Token",
        value="****" if has_token else "",
        type="password",
        disabled=True,
        help="Set TELEGRAM_BOT_TOKEN in .env file",
    )

    st.text_input(
        "Chat ID",
        value=settings.telegram.chat_id or "",
        disabled=True,
        help="Set TELEGRAM_CHAT_ID in .env file",
    )

    st.subheader("Alert Preferences")

    col1, col2 = st.columns(2)

    with col1:
        st.multiselect(
            "Markets to Alert",
            ["1X2", "Over/Under", "BTTS"],
            default=["1X2", "Over/Under", "BTTS"],
            key="alert_markets",
            disabled=True,
        )

    with col2:
        st.multiselect(
            "Alert Timing",
            ["D-7", "D-3", "D-1", "H-2"],
            default=["D-1", "H-2"],
            key="alert_timing",
            disabled=True,
        )

    if st.button("Test Telegram Connection"):
        if has_token and settings.telegram.chat_id:
            st.info("Telegram test functionality will be available in Phase 5")
        else:
            st.warning("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first")

    st.info("Alert settings are read from environment variables. Edit the .env file and restart to change.")
