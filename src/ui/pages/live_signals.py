"""Live signals page - shows current edge opportunities."""
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.storage.database import db
from src.storage.models import Match, Prediction


# Edge color constants for fintech theme
EDGE_COLORS = {
    "strong": {"hex": "#10B981", "class": "edge-strong", "label": "Strong"},
    "moderate": {"hex": "#F59E0B", "class": "edge-moderate", "label": "Moderate"},
    "marginal": {"hex": "#FB923C", "class": "edge-marginal", "label": "Marginal"},
    "below": {"hex": "#64748B", "class": "edge-below", "label": "Below"},
}


def _get_edge_tier(edge: float) -> dict:
    """Get edge tier info based on value."""
    if edge >= 0.10:
        return EDGE_COLORS["strong"]
    elif edge >= 0.07:
        return EDGE_COLORS["moderate"]
    elif edge >= 0.05:
        return EDGE_COLORS["marginal"]
    return EDGE_COLORS["below"]


def render():
    """Render live signals page."""
    st.header("Live Signals")
    st.caption("Real-time edge detection across EPL markets")

    # Filters in styled container
    st.markdown("#### Filters")
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

    st.divider()

    # Fetch and display signals
    signals = _get_signals(market_filter, min_edge / 100, time_range)

    if not signals:
        st.info("No signals matching your criteria. Try adjusting filters or check back later.")
        return

    # Summary metrics
    _render_summary_metrics(signals)

    st.markdown("#### Active Signals")
    _render_signal_cards(signals)


def _render_summary_metrics(signals: list):
    """Render summary metrics for signals."""
    total = len(signals)
    strong = sum(1 for s in signals if s["edge"] >= 0.10)
    moderate = sum(1 for s in signals if 0.07 <= s["edge"] < 0.10)
    avg_edge = sum(s["edge"] for s in signals) / total if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", total)
    with col2:
        st.metric("Strong Edges", strong, help="Edge >= 10%")
    with col3:
        st.metric("Moderate Edges", moderate, help="Edge 7-10%")
    with col4:
        st.metric("Avg Edge", f"{avg_edge * 100:.1f}%")

    st.divider()


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
        edge_tier = _get_edge_tier(signal["edge"])

        # Create styled signal card with HTML
        card_html = f"""
        <div class="signal-card {edge_tier['class']}" style="
            background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-left: 4px solid {edge_tier['hex']};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-family: 'Poppins', sans-serif; font-size: 1.1rem; font-weight: 600; color: #F8FAFC;">
                    {match.home_team} vs {match.away_team}
                </div>
                <div style="
                    background: {edge_tier['hex']}20;
                    color: {edge_tier['hex']};
                    padding: 0.25rem 0.75rem;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    text-transform: uppercase;
                ">{edge_tier['label']} Edge</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market", signal["market"])
        with col2:
            st.metric("Selection", signal["selection"])
        with col3:
            # Color-coded edge value
            edge_pct = signal['edge'] * 100
            st.metric("Edge", f"{edge_pct:.1f}%")
        with col4:
            st.metric("Fair Price", f"{signal['fair_prob'] * 100:.1f}%")

        # Model source indicator
        model_source = signal.get("model_source", "poisson")
        ml_confidence = signal.get("ml_confidence")
        _render_model_source(model_source, ml_confidence)

        # Details expander
        with st.expander("View Details"):
            kickoff_str = match.kickoff.strftime("%Y-%m-%d %H:%M UTC") if match.kickoff else "N/A"

            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown("**Match Info**")
                st.write(f"Kickoff: {kickoff_str}")
                st.write(f"Weight Profile: {signal['preset']}")
            with detail_col2:
                st.markdown("**Probability Comparison**")
                st.write(f"Model: {signal['fair_prob'] * 100:.1f}%")
                st.write(f"Market: {signal['market_prob'] * 100:.1f}%")

            # Show hybrid details if available
            if signal.get("poisson_prob") and signal.get("ml_prob"):
                st.divider()
                st.markdown("**Model Breakdown**")
                hybrid_col1, hybrid_col2, hybrid_col3 = st.columns(3)
                with hybrid_col1:
                    st.write(f"Poisson: {signal['poisson_prob'] * 100:.1f}%")
                with hybrid_col2:
                    st.write(f"ML: {signal['ml_prob'] * 100:.1f}%")
                with hybrid_col3:
                    agreement = signal.get("agreement_score", 1.0)
                    st.write(f"Agreement: {agreement * 100:.0f}%")

        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


def _render_model_source(model_source: str, ml_confidence: Optional[str] = None):
    """Render model source indicator with styled badge."""
    if model_source == "hybrid":
        conf_display = f" ({ml_confidence})" if ml_confidence else ""
        badge_html = f"""
        <div style="display: inline-block; background: #8B5CF620; color: #8B5CF6; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-top: 0.5rem;">
            Hybrid Model{conf_display}
        </div>
        """
    elif model_source == "ml":
        badge_html = """
        <div style="display: inline-block; background: #06B6D420; color: #06B6D4; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-top: 0.5rem;">
            ML Classifier
        </div>
        """
    else:
        badge_html = """
        <div style="display: inline-block; background: #64748B20; color: #94A3B8; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-top: 0.5rem;">
            Poisson Model
        </div>
        """
    st.markdown(badge_html, unsafe_allow_html=True)
