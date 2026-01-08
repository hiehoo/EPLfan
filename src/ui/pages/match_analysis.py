"""Detailed match analysis page."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from src.storage.database import db
from src.storage.models import Match, OddsSnapshot, TeamStats
from src.models import match_analyzer


def render():
    """Render match analysis page."""
    st.header("Match Analysis")

    # Match selector
    matches = _get_upcoming_matches()
    if not matches:
        st.warning("No upcoming matches found. Add matches to the database first.")
        _render_demo_analysis()
        return

    match_options = {
        f"{m['home_team']} vs {m['away_team']} ({m['kickoff'].strftime('%d/%m %H:%M')})": m
        for m in matches
    }
    selected = st.selectbox("Select Match", list(match_options.keys()))
    match = match_options[selected]

    # Get data and analyze
    analysis_result = _analyze_match(match["id"])

    if not analysis_result:
        st.error("Failed to analyze match - missing data")
        return

    match_probs, weighted_probs, edge_analysis = analysis_result

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1X2", "Over/Under", "BTTS", "Poisson Matrix", "ML Insights"])

    with tab1:
        _render_1x2_analysis(match_probs, weighted_probs, edge_analysis)

    with tab2:
        _render_ou_analysis(match_probs, weighted_probs, edge_analysis)

    with tab3:
        _render_btts_analysis(match_probs, weighted_probs, edge_analysis)

    with tab4:
        _render_poisson_matrix(match_probs)

    with tab5:
        _render_ml_insights(edge_analysis)


def _get_upcoming_matches() -> list[dict]:
    """Fetch upcoming matches as dictionaries to avoid session detachment."""
    with db.session() as session:
        now = datetime.now(timezone.utc)
        matches = (
            session.query(Match)
            .filter(Match.kickoff >= now, Match.is_completed == False)
            .order_by(Match.kickoff)
            .limit(20)
            .all()
        )
        # Convert to dicts to avoid DetachedInstanceError
        return [
            {
                "id": m.id,
                "external_id": m.external_id,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "kickoff": m.kickoff,
            }
            for m in matches
        ]


def _validate_xg(value, default: float = 1.0) -> float:
    """Validate xG is in reasonable range."""
    if value is not None and 0 < value < 10:
        return value
    return default

def _validate_elo(value, default: float = 1500.0) -> float:
    """Validate ELO is in reasonable range."""
    if value is not None and 1000 < value < 3000:
        return value
    return default


def _analyze_match(match_id: int) -> Optional[tuple]:
    """Run full analysis on a match by ID."""
    with db.session() as session:
        # Get match
        match = session.query(Match).filter_by(id=match_id).first()
        if not match:
            return None

        # Get latest odds snapshot
        odds = (
            session.query(OddsSnapshot)
            .filter(OddsSnapshot.match_id == match_id)
            .order_by(OddsSnapshot.snapshot_time.desc())
            .first()
        )

        if not odds:
            return None

        # Get team stats
        home_stats = (
            session.query(TeamStats)
            .filter(TeamStats.team_name == match.home_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )
        away_stats = (
            session.query(TeamStats)
            .filter(TeamStats.team_name == match.away_team)
            .order_by(TeamStats.snapshot_date.desc())
            .first()
        )

        if not home_stats or not away_stats:
            return None

        # Check if ML features available
        is_elite = getattr(home_stats, 'is_elite', False) or getattr(away_stats, 'is_elite', False)

        # Run analysis with ML features if available
        try:
            return match_analyzer.analyze(
                match_id=str(match_id),
                home_team=match.home_team,
                away_team=match.away_team,
                kickoff=match.kickoff,
                home_xg=_validate_xg(home_stats.home_xg),
                away_xg=_validate_xg(away_stats.away_xg),
                home_xga=_validate_xg(home_stats.home_xga),
                away_xga=_validate_xg(away_stats.away_xga),
                home_elo=_validate_elo(home_stats.elo_rating),
                away_elo=_validate_elo(away_stats.elo_rating),
                bf_home_odds=odds.bf_home_odds or 2.5,
                bf_draw_odds=odds.bf_draw_odds or 3.5,
                bf_away_odds=odds.bf_away_odds or 3.0,
                bf_over_2_5_odds=odds.bf_over_2_5_odds,
                bf_under_2_5_odds=odds.bf_under_2_5_odds,
                bf_btts_yes_odds=odds.bf_btts_yes_odds,
                bf_btts_no_odds=odds.bf_btts_no_odds,
                pm_home_price=odds.pm_home_price,
                pm_draw_price=odds.pm_draw_price,
                pm_away_price=odds.pm_away_price,
                pm_over_2_5_price=odds.pm_over_2_5_price,
                pm_under_2_5_price=odds.pm_under_2_5_price,
                pm_btts_yes_price=odds.pm_btts_yes_price,
                pm_btts_no_price=odds.pm_btts_no_price,
                # ML feature stats
                home_shots_pg=getattr(home_stats, 'shots_per_game', None),
                away_shots_pg=getattr(away_stats, 'shots_per_game', None),
                home_volatility=getattr(home_stats, 'goal_volatility', None),
                away_volatility=getattr(away_stats, 'goal_volatility', None),
                home_conversion=getattr(home_stats, 'shot_conversion_rate', None),
                away_conversion=getattr(away_stats, 'shot_conversion_rate', None),
                is_elite_match=is_elite,
            )
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return None


def _render_demo_analysis():
    """Render demo analysis when no matches available."""
    st.info("Showing demo analysis with sample data")

    # Demo data
    from src.models import match_analyzer as ma

    result = ma.analyze(
        match_id="demo",
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff=datetime.now(timezone.utc),
        home_xg=1.8,
        away_xg=1.4,
        home_xga=1.0,
        away_xga=1.2,
        home_elo=2050,
        away_elo=1980,
        bf_home_odds=2.1,
        bf_draw_odds=3.5,
        bf_away_odds=3.4,
        bf_over_2_5_odds=1.9,
        bf_under_2_5_odds=2.0,
        bf_btts_yes_odds=1.8,
        bf_btts_no_odds=2.1,
        pm_home_price=0.45,
        pm_draw_price=0.28,
        pm_away_price=0.27,
        pm_over_2_5_price=0.55,
        pm_under_2_5_price=0.45,
        pm_btts_yes_price=0.58,
        pm_btts_no_price=0.42,
    )

    match_probs, weighted_probs, edge_analysis = result

    tab1, tab2, tab3, tab4 = st.tabs(["1X2", "Over/Under", "BTTS", "Poisson Matrix"])

    with tab1:
        _render_1x2_analysis(match_probs, weighted_probs, edge_analysis)
    with tab2:
        _render_ou_analysis(match_probs, weighted_probs, edge_analysis)
    with tab3:
        _render_btts_analysis(match_probs, weighted_probs, edge_analysis)
    with tab4:
        _render_poisson_matrix(match_probs)


def _render_1x2_analysis(match_probs, weighted_probs, edge_analysis):
    """Render 1X2 market analysis."""
    st.subheader("1X2 Market")

    # Bar chart comparing sources
    fig = go.Figure()

    sources = ["Betfair", "xG Model", "ELO", "Fair Price", "Polymarket"]
    home_probs = [
        match_probs.betfair_1x2.get("home", 0),
        match_probs.poisson_result.p_1x2.get("home", 0),
        match_probs.elo_1x2.get("home", 0),
        weighted_probs.fair_1x2.get("home", 0),
        match_probs.polymarket_1x2.get("home", 0),
    ]
    draw_probs = [
        match_probs.betfair_1x2.get("draw", 0),
        match_probs.poisson_result.p_1x2.get("draw", 0),
        match_probs.elo_1x2.get("draw", 0),
        weighted_probs.fair_1x2.get("draw", 0),
        match_probs.polymarket_1x2.get("draw", 0),
    ]
    away_probs = [
        match_probs.betfair_1x2.get("away", 0),
        match_probs.poisson_result.p_1x2.get("away", 0),
        match_probs.elo_1x2.get("away", 0),
        weighted_probs.fair_1x2.get("away", 0),
        match_probs.polymarket_1x2.get("away", 0),
    ]

    fig.add_trace(go.Bar(name="Home", x=sources, y=home_probs, marker_color="#28a745"))
    fig.add_trace(go.Bar(name="Draw", x=sources, y=draw_probs, marker_color="#6c757d"))
    fig.add_trace(go.Bar(name="Away", x=sources, y=away_probs, marker_color="#dc3545"))

    fig.update_layout(
        barmode="group",
        title="1X2 Probability Comparison",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
    )

    st.plotly_chart(fig, width="stretch")

    # Edge display
    col1, col2, col3 = st.columns(3)
    edges = edge_analysis.all_edges.get("1x2", {})

    with col1:
        _edge_metric(
            "Home Win",
            weighted_probs.fair_1x2.get("home", 0),
            edges.get("home", {}).get("edge", 0),
        )
    with col2:
        _edge_metric(
            "Draw",
            weighted_probs.fair_1x2.get("draw", 0),
            edges.get("draw", {}).get("edge", 0),
        )
    with col3:
        _edge_metric(
            "Away Win",
            weighted_probs.fair_1x2.get("away", 0),
            edges.get("away", {}).get("edge", 0),
        )


def _render_ou_analysis(match_probs, weighted_probs, edge_analysis):
    """Render Over/Under analysis."""
    st.subheader("Over/Under 2.5 Goals")

    fair_ou = weighted_probs.fair_ou.get("2.5", {})
    edges = edge_analysis.all_edges.get("ou", {})

    col1, col2 = st.columns(2)
    with col1:
        _edge_metric(
            "Over 2.5",
            fair_ou.get("over", 0),
            edges.get("over", {}).get("edge", 0),
        )
    with col2:
        _edge_metric(
            "Under 2.5",
            fair_ou.get("under", 0),
            edges.get("under", {}).get("edge", 0),
        )

    # Show all goal line probabilities from Poisson
    st.markdown("#### Goal Line Probabilities (Poisson)")
    poisson = match_probs.poisson_result

    ou_data = []
    for line in ["1.5", "2.5", "3.5", "4.5"]:
        ou = poisson.p_over_under.get(line, {})
        ou_data.append({
            "Line": line,
            "Over": f"{ou.get('over', 0) * 100:.1f}%",
            "Under": f"{ou.get('under', 0) * 100:.1f}%",
        })

    st.table(ou_data)


def _render_btts_analysis(match_probs, weighted_probs, edge_analysis):
    """Render BTTS analysis."""
    st.subheader("Both Teams to Score")

    edges = edge_analysis.all_edges.get("btts", {})

    col1, col2 = st.columns(2)
    with col1:
        _edge_metric(
            "BTTS Yes",
            weighted_probs.fair_btts.get("yes", 0),
            edges.get("yes", {}).get("edge", 0),
        )
    with col2:
        _edge_metric(
            "BTTS No",
            weighted_probs.fair_btts.get("no", 0),
            edges.get("no", {}).get("edge", 0),
        )


def _render_poisson_matrix(match_probs):
    """Render Poisson probability matrix."""
    st.subheader("Scoreline Probabilities")

    matrix = match_probs.poisson_result.matrix

    # Create labels for display
    labels = [str(i) for i in range(matrix.shape[0])]

    fig = px.imshow(
        matrix,
        labels=dict(x="Away Goals", y="Home Goals", color="Probability"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=".1%",
    )
    fig.update_layout(title="Poisson Probability Matrix")

    st.plotly_chart(fig, width="stretch")

    # Most likely scorelines
    st.markdown("**Most Likely Scorelines:**")
    scorelines = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            scorelines.append((f"{i}-{j}", matrix[i, j]))
    scorelines.sort(key=lambda x: x[1], reverse=True)

    cols = st.columns(5)
    for idx, (score, prob) in enumerate(scorelines[:5]):
        with cols[idx]:
            st.metric(score, f"{prob * 100:.1f}%")


def _edge_metric(label: str, fair_prob: float, edge: float):
    """Display edge metric with color coding."""
    if edge > 0.07:
        delta_color = "normal"
        prefix = "🟢"
    elif edge > 0.05:
        delta_color = "normal"
        prefix = "🟡"
    elif edge > 0:
        delta_color = "off"
        prefix = ""
    else:
        delta_color = "inverse"
        prefix = ""

    st.metric(
        f"{prefix} {label}",
        f"{fair_prob * 100:.1f}%",
        delta=f"{edge * 100:+.1f}% edge",
        delta_color=delta_color,
    )


def _has_valid_ml_prob(signal) -> bool:
    """Check if signal has valid ML probability."""
    if not signal:
        return False
    try:
        ml_prob = getattr(signal, "ml_prob", None)
        return ml_prob is not None and isinstance(ml_prob, (int, float))
    except Exception:
        return False


def _render_ml_insights(edge_analysis):
    """Render ML model insights."""
    st.subheader("ML Classification Insights")

    # Check if ML data available
    signal_ou = edge_analysis.signal_ou
    signal_btts = edge_analysis.signal_btts

    has_ml_ou = _has_valid_ml_prob(signal_ou)
    has_ml_btts = _has_valid_ml_prob(signal_btts)

    if not has_ml_ou and not has_ml_btts:
        st.info("ML models not trained or no ML data available for this match. "
                "Run `python -m src.cli train-models` to train ML classifiers.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Over/Under 2.5**")
        if has_ml_ou:
            poisson_prob = getattr(signal_ou, "poisson_prob", None)
            if poisson_prob is None:
                poisson_prob = getattr(signal_ou, "fair_prob", 0.5)
            ml_prob = getattr(signal_ou, "ml_prob", 0.5)
            ml_conf = getattr(signal_ou, "ml_confidence", "unknown")

            # Show comparison
            st.metric(
                label="Poisson Model",
                value=f"{poisson_prob:.1%}",
                help="Traditional Poisson-based probability",
            )
            st.metric(
                label=f"ML Ensemble ({ml_conf})",
                value=f"{ml_prob:.1%}",
                delta=f"{(ml_prob - poisson_prob) * 100:+.1f}% vs Poisson",
            )
            st.metric(
                label="Hybrid (Blended)",
                value=f"{signal_ou.fair_prob:.1%}",
            )
        else:
            st.caption("No ML prediction available")

    with col2:
        st.markdown("**Both Teams to Score**")
        if has_ml_btts:
            poisson_prob = getattr(signal_btts, "poisson_prob", None)
            if poisson_prob is None:
                poisson_prob = getattr(signal_btts, "fair_prob", 0.5)
            ml_prob = getattr(signal_btts, "ml_prob", 0.5)
            ml_conf = getattr(signal_btts, "ml_confidence", "unknown")

            st.metric(
                label="Poisson Model",
                value=f"{poisson_prob:.1%}",
            )
            st.metric(
                label=f"ML Ensemble ({ml_conf})",
                value=f"{ml_prob:.1%}",
                delta=f"{(ml_prob - poisson_prob) * 100:+.1f}% vs Poisson",
            )
            st.metric(
                label="Hybrid (Blended)",
                value=f"{signal_btts.fair_prob:.1%}",
            )
        else:
            st.caption("No ML prediction available")

    # Agreement indicator
    if has_ml_ou:
        agreement = getattr(signal_ou, "agreement_score", 1.0)
        if agreement > 0.9:
            st.success(f"Models strongly agree ({agreement:.0%})")
        elif agreement > 0.7:
            st.warning(f"Models partially agree ({agreement:.0%})")
        else:
            st.error(f"Models disagree significantly ({agreement:.0%})")

    # Model source info
    st.markdown("---")
    st.caption("**Model Info:** Hybrid predictions blend Poisson (statistical) and ML "
              "(LightGBM/XGBoost/RandomForest ensemble) based on ML confidence level.")
