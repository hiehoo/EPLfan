"""Historical performance tracking page."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from typing import Optional

from src.storage.database import db
from src.storage.models import Prediction, Match


def render():
    """Render historical performance page."""
    st.header("Historical Performance")

    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Load data
    predictions = _get_historical_predictions(start_date, end_date)

    if predictions.empty:
        st.info("No historical data in selected range. Predictions will appear here after matches are resolved.")
        _render_demo_historical()
        return

    # Summary metrics
    _render_summary_metrics(predictions)

    # Performance by market
    st.subheader("Performance by Market")
    _render_market_performance(predictions)

    # Calibration chart
    st.subheader("Calibration")
    _render_calibration_chart(predictions)

    # Edge vs Accuracy scatter
    st.subheader("Edge vs Hit Rate")
    _render_edge_accuracy(predictions)


def _get_historical_predictions(start_date, end_date) -> pd.DataFrame:
    """Fetch resolved predictions."""
    # Make datetime timezone-aware (UTC)
    start_dt = datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc)

    with db.session() as session:
        results = (
            session.query(
                Prediction.preset_used,
                Prediction.p_home_model,
                Prediction.p_draw_model,
                Prediction.p_away_model,
                Prediction.p_home_market,
                Prediction.edge_1x2,
                Prediction.signal_1x2,
                Prediction.is_1x2_correct,
                Prediction.p_over_2_5_model,
                Prediction.p_over_2_5_market,
                Prediction.edge_ou,
                Prediction.signal_ou,
                Prediction.is_ou_correct,
                Prediction.p_btts_yes_model,
                Prediction.p_btts_yes_market,
                Prediction.edge_btts,
                Prediction.signal_btts,
                Prediction.is_btts_correct,
                Match.kickoff,
                Match.home_team,
                Match.away_team,
            )
            .join(Match)
            .filter(
                Match.kickoff >= start_dt,
                Match.kickoff <= end_dt,
                Match.is_completed == True,
            )
            .all()
        )

        if not results:
            return pd.DataFrame()

        # Build dataframe with resolved predictions
        data = []
        for row in results:
            # 1X2 prediction
            if row.signal_1x2 and row.signal_1x2 != "none" and row.is_1x2_correct is not None:
                fair_prob = {
                    "home": row.p_home_model,
                    "draw": row.p_draw_model,
                    "away": row.p_away_model,
                }.get(row.signal_1x2, 0)

                data.append({
                    "market_type": "1x2",
                    "selection": row.signal_1x2,
                    "fair_prob": fair_prob,
                    "market_prob": row.p_home_market,
                    "edge": abs(row.edge_1x2),
                    "preset": row.preset_used,
                    "outcome": 1 if row.is_1x2_correct else 0,
                    "kickoff": row.kickoff,
                })

            # O/U prediction
            if row.signal_ou and row.signal_ou != "none" and row.is_ou_correct is not None:
                fair_prob = row.p_over_2_5_model if row.signal_ou == "over" else (1 - row.p_over_2_5_model)

                data.append({
                    "market_type": "ou",
                    "selection": row.signal_ou,
                    "fair_prob": fair_prob,
                    "market_prob": row.p_over_2_5_market,
                    "edge": abs(row.edge_ou),
                    "preset": row.preset_used,
                    "outcome": 1 if row.is_ou_correct else 0,
                    "kickoff": row.kickoff,
                })

            # BTTS prediction
            if row.signal_btts and row.signal_btts != "none" and row.is_btts_correct is not None:
                fair_prob = row.p_btts_yes_model if row.signal_btts == "yes" else (1 - row.p_btts_yes_model)

                data.append({
                    "market_type": "btts",
                    "selection": row.signal_btts,
                    "fair_prob": fair_prob,
                    "market_prob": row.p_btts_yes_market,
                    "edge": abs(row.edge_btts),
                    "preset": row.preset_used,
                    "outcome": 1 if row.is_btts_correct else 0,
                    "kickoff": row.kickoff,
                })

        return pd.DataFrame(data)


def _render_demo_historical():
    """Render demo historical data."""
    st.info("Showing demo data for illustration")

    # Create sample data
    import numpy as np
    np.random.seed(42)

    n = 100
    demo_df = pd.DataFrame({
        "market_type": np.random.choice(["1x2", "ou", "btts"], n),
        "selection": np.random.choice(["home", "over", "yes"], n),
        "fair_prob": np.random.uniform(0.4, 0.7, n),
        "market_prob": np.random.uniform(0.35, 0.65, n),
        "edge": np.random.uniform(0.03, 0.15, n),
        "preset": np.random.choice(["market_trust", "balanced", "analytics_first"], n),
        "outcome": np.random.choice([0, 1], n, p=[0.45, 0.55]),
    })

    _render_summary_metrics(demo_df)
    st.subheader("Performance by Market")
    _render_market_performance(demo_df)
    st.subheader("Calibration")
    _render_calibration_chart(demo_df)
    st.subheader("Edge vs Hit Rate")
    _render_edge_accuracy(demo_df)


def _render_summary_metrics(df: pd.DataFrame):
    """Render summary performance metrics."""
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    wins = df["outcome"].sum()
    hit_rate = wins / total if total > 0 else 0

    # Expected hit rate based on fair probabilities
    expected_rate = df["fair_prob"].mean()

    with col1:
        st.metric("Total Predictions", total)
    with col2:
        st.metric("Wins", int(wins))
    with col3:
        st.metric(
            "Hit Rate",
            f"{hit_rate * 100:.1f}%",
            delta=f"{(hit_rate - expected_rate) * 100:+.1f}% vs expected",
        )
    with col4:
        # Brier score
        brier = ((df["fair_prob"] - df["outcome"].astype(float)) ** 2).mean()
        st.metric("Brier Score", f"{brier:.3f}")


def _render_market_performance(df: pd.DataFrame):
    """Render performance breakdown by market."""
    if df.empty:
        st.info("No data available for market performance analysis")
        return

    market_stats = (
        df.groupby("market_type")
        .agg({
            "outcome": ["sum", "count", "mean"],
            "fair_prob": "mean",
            "edge": "mean",
        })
        .round(3)
    )

    market_stats.columns = ["Wins", "Total", "Hit Rate", "Avg Fair Prob", "Avg Edge"]
    market_stats["Expected"] = market_stats["Avg Fair Prob"]

    # Format for display
    display_df = market_stats.copy()
    display_df["Hit Rate"] = display_df["Hit Rate"].apply(lambda x: f"{x:.1%}")
    display_df["Expected"] = display_df["Expected"].apply(lambda x: f"{x:.1%}")
    display_df["Avg Edge"] = display_df["Avg Edge"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df)


def _render_calibration_chart(df: pd.DataFrame):
    """Render calibration chart - predicted vs actual."""
    if df.empty:
        st.info("No data available for calibration analysis")
        return

    # Bin predictions by probability
    df_cal = df.copy()
    df_cal["prob_bin"] = pd.cut(
        df_cal["fair_prob"],
        bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["0-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-100%"],
    )

    calibration = (
        df_cal.groupby("prob_bin", observed=True)
        .agg({
            "outcome": "mean",
            "fair_prob": "mean",
        })
        .reset_index()
    )

    # Check sample size per bin
    bin_counts = df_cal.groupby("prob_bin", observed=True).size()
    if len(bin_counts) > 0 and bin_counts.max() < 5:
        st.warning("⚠️ Low sample size per bin (<5). Calibration results may be unreliable.")

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        )
    )

    # Actual calibration
    if not calibration.empty:
        fig.add_trace(
            go.Scatter(
                x=calibration["fair_prob"],
                y=calibration["outcome"],
                mode="markers+lines",
                name="Actual",
                marker=dict(size=10),
            )
        )

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Outcome Rate",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_edge_accuracy(df: pd.DataFrame):
    """Render edge vs accuracy scatter."""
    if df.empty:
        st.info("No data available for edge accuracy analysis")
        return

    # Group by edge buckets
    df_edge = df.copy()
    df_edge["edge_bucket"] = pd.cut(
        df_edge["edge"],
        bins=[0, 0.05, 0.07, 0.10, 0.15, 1.0],
        labels=["0-5%", "5-7%", "7-10%", "10-15%", "15%+"],
    )

    edge_perf = (
        df_edge.groupby("edge_bucket", observed=True)
        .agg({
            "outcome": ["mean", "count"],
        })
        .reset_index()
    )
    edge_perf.columns = ["edge_bucket", "hit_rate", "count"]

    fig = px.bar(
        edge_perf,
        x="edge_bucket",
        y="hit_rate",
        text="count",
        title="Hit Rate by Edge Size",
    )
    fig.update_layout(
        xaxis_title="Edge Bucket",
        yaxis_title="Hit Rate",
        yaxis_tickformat=".0%",
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
