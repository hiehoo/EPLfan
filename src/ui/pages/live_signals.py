"""Live signals page - shows current edge opportunities."""
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.storage.database import db
from src.storage.models import Match, Prediction


def render():
    """Render live signals page."""
    st.header("Live Signals")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        market_filter = st.selectbox(
            "Market",
            ["All", "1X2", "Over/Under", "BTTS"],
        )
    with col2:
        min_edge = st.slider("Min Edge %", 0.0, 20.0, 5.0, 0.5)
    with col3:
        time_range = st.selectbox(
            "Time Range",
            ["Next 24h", "Next 48h", "Next 7 days"],
        )

    # Fetch and display signals
    signals = _get_signals(market_filter, min_edge / 100, time_range)

    if not signals:
        st.info("No signals matching criteria")
        return

    _render_signal_cards(signals)


def _get_signals(market_filter: str, min_edge: float, time_range: str) -> list:
    """Fetch signals from database."""
    with db.session() as session:
        now = datetime.now(timezone.utc)

        if time_range == "Next 24h":
            cutoff = now + timedelta(hours=24)
        elif time_range == "Next 48h":
            cutoff = now + timedelta(hours=48)
        else:
            cutoff = now + timedelta(days=7)

        # Query matches with predictions
        query = (
            session.query(Prediction, Match)
            .join(Match)
            .filter(
                Match.kickoff >= now,
                Match.kickoff <= cutoff,
                Match.is_completed == False,
            )
        )

        results = query.all()
        signals = []

        for pred, match in results:
            # Check each market type based on filter
            if market_filter in ["All", "1X2"]:
                if pred.signal_1x2 and pred.signal_1x2 != "none" and abs(pred.edge_1x2) >= min_edge:
                    signals.append({
                        "match": match,
                        "market": "1X2",
                        "selection": pred.signal_1x2.upper(),
                        "edge": abs(pred.edge_1x2),
                        "fair_prob": _get_fair_prob(pred, "1x2"),
                        "market_prob": _get_market_prob(pred, "1x2"),
                        "preset": pred.preset_used,
                    })

            if market_filter in ["All", "Over/Under"]:
                if pred.signal_ou and pred.signal_ou != "none" and abs(pred.edge_ou) >= min_edge:
                    signals.append({
                        "match": match,
                        "market": "O/U 2.5",
                        "selection": pred.signal_ou.upper(),
                        "edge": abs(pred.edge_ou),
                        "fair_prob": pred.p_over_2_5_model if pred.signal_ou == "over" else (1 - pred.p_over_2_5_model),
                        "market_prob": pred.p_over_2_5_market if pred.signal_ou == "over" else (1 - pred.p_over_2_5_market),
                        "preset": pred.preset_used,
                    })

            if market_filter in ["All", "BTTS"]:
                if pred.signal_btts and pred.signal_btts != "none" and abs(pred.edge_btts) >= min_edge:
                    signals.append({
                        "match": match,
                        "market": "BTTS",
                        "selection": pred.signal_btts.upper(),
                        "edge": abs(pred.edge_btts),
                        "fair_prob": pred.p_btts_yes_model if pred.signal_btts == "yes" else (1 - pred.p_btts_yes_model),
                        "market_prob": pred.p_btts_yes_market if pred.signal_btts == "yes" else (1 - pred.p_btts_yes_market),
                        "preset": pred.preset_used,
                    })

        # Sort by edge descending
        signals.sort(key=lambda x: x["edge"], reverse=True)
        return signals


def _get_fair_prob(pred: Prediction, market: str) -> float:
    """Get fair probability for the signal direction."""
    if market == "1x2":
        if pred.signal_1x2 == "home":
            return pred.p_home_model
        elif pred.signal_1x2 == "draw":
            return pred.p_draw_model
        elif pred.signal_1x2 == "away":
            return pred.p_away_model
    return 0.0


def _get_market_prob(pred: Prediction, market: str) -> float:
    """Get market probability for the signal direction."""
    if market == "1x2":
        if pred.signal_1x2 == "home":
            return pred.p_home_market
        elif pred.signal_1x2 == "draw":
            return pred.p_draw_market
        elif pred.signal_1x2 == "away":
            return pred.p_away_market
    return 0.0


def _render_signal_cards(signals: list):
    """Render signal cards with edge visualization."""
    for signal in signals:
        match = signal["match"]

        # Color coding based on edge strength
        if signal["edge"] >= 0.10:
            edge_color = "🟢"  # Strong
        elif signal["edge"] >= 0.07:
            edge_color = "🟡"  # Moderate
        else:
            edge_color = "🟠"  # Marginal

        with st.container():
            st.markdown(f"### {edge_color} {match.home_team} vs {match.away_team}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market", signal["market"])
            with col2:
                st.metric("Selection", signal["selection"])
            with col3:
                st.metric("Edge", f"{signal['edge'] * 100:.1f}%")
            with col4:
                st.metric("Fair Price", f"{signal['fair_prob'] * 100:.1f}%")

            # Details expander
            with st.expander("Details"):
                kickoff_str = match.kickoff.strftime("%Y-%m-%d %H:%M") if match.kickoff else "N/A"
                st.write(f"**Kickoff:** {kickoff_str}")
                st.write(f"**Polymarket:** {signal['market_prob'] * 100:.1f}%")
                st.write(f"**Weight Profile:** {signal['preset']}")

            st.divider()
