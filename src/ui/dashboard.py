"""EPL Bet Indicator Dashboard."""
import streamlit as st

st.set_page_config(
    page_title="EPL Bet Indicator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main dashboard entry."""
    st.title("⚽ EPL Bet Indicator v2")
    st.caption("Multi-market betting indicator for Polymarket")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Live Signals", "Match Analysis", "Historical Performance", "Settings"],
    )

    if page == "Live Signals":
        from src.ui.pages import live_signals
        live_signals.render()
    elif page == "Match Analysis":
        from src.ui.pages import match_analysis
        match_analysis.render()
    elif page == "Historical Performance":
        from src.ui.pages import historical
        historical.render()
    elif page == "Settings":
        from src.ui.pages import settings_page
        settings_page.render()


if __name__ == "__main__":
    main()
