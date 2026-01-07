"""Database models for EPL Bet Indicator."""
from datetime import datetime
from typing import Optional

from sqlalchemy import Float, String, DateTime, Integer, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Match(Base):
    """EPL match record."""
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True)
    home_team: Mapped[str] = mapped_column(String(100))
    away_team: Mapped[str] = mapped_column(String(100))
    kickoff: Mapped[datetime] = mapped_column(DateTime)

    # Actual result (filled post-match)
    home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    odds_snapshots: Mapped[list["OddsSnapshot"]] = relationship(back_populates="match")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="match")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OddsSnapshot(Base):
    """Point-in-time odds snapshot for a match."""
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    snapshot_time: Mapped[datetime] = mapped_column(DateTime)
    hours_to_kickoff: Mapped[float] = mapped_column(Float)

    # Betfair odds
    bf_home_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_draw_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_away_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_over_2_5_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_under_2_5_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_btts_yes_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_btts_no_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Polymarket prices
    pm_home_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_draw_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_away_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_over_2_5_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_under_2_5_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_btts_yes_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_btts_no_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Liquidity
    pm_home_liquidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_over_2_5_liquidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pm_btts_yes_liquidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="odds_snapshots")


class TeamStats(Base):
    """Team statistics snapshot."""
    __tablename__ = "team_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_name: Mapped[str] = mapped_column(String(100))
    snapshot_date: Mapped[datetime] = mapped_column(DateTime)

    # xG data
    xg_for: Mapped[float] = mapped_column(Float)
    xga_against: Mapped[float] = mapped_column(Float)
    xpts: Mapped[float] = mapped_column(Float)
    home_xg: Mapped[float] = mapped_column(Float)
    away_xg: Mapped[float] = mapped_column(Float)
    home_xga: Mapped[float] = mapped_column(Float)
    away_xga: Mapped[float] = mapped_column(Float)

    # ELO
    elo_rating: Mapped[float] = mapped_column(Float)
    elo_rank: Mapped[int] = mapped_column(Integer)

    # Shot data (Phase 2)
    shots_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shots_against_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shots_on_target_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Volatility (Phase 2)
    goal_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xg_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Efficiency (Phase 2)
    shot_conversion_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xg_overperformance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Elite team tracking
    is_elite: Mapped[bool] = mapped_column(Boolean, default=False)
    elite_adjustment: Mapped[float] = mapped_column(Float, default=1.0)


class Prediction(Base):
    """Model prediction for a match."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    preset_used: Mapped[str] = mapped_column(String(50))  # market_trust, balanced, analytics_first

    # 1X2 predictions
    p_home_model: Mapped[float] = mapped_column(Float)
    p_draw_model: Mapped[float] = mapped_column(Float)
    p_away_model: Mapped[float] = mapped_column(Float)
    p_home_market: Mapped[float] = mapped_column(Float)
    p_draw_market: Mapped[float] = mapped_column(Float)
    p_away_market: Mapped[float] = mapped_column(Float)
    edge_1x2: Mapped[float] = mapped_column(Float)
    signal_1x2: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # home, draw, away, none

    # O/U predictions
    p_over_2_5_model: Mapped[float] = mapped_column(Float)
    p_over_2_5_market: Mapped[float] = mapped_column(Float)
    edge_ou: Mapped[float] = mapped_column(Float)
    signal_ou: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # over, under, none

    # BTTS predictions
    p_btts_yes_model: Mapped[float] = mapped_column(Float)
    p_btts_yes_market: Mapped[float] = mapped_column(Float)
    edge_btts: Mapped[float] = mapped_column(Float)
    signal_btts: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # yes, no, none

    # Outcome tracking (filled post-match)
    is_1x2_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_ou_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_btts_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="predictions")


class HistoricalOutcome(Base):
    """Track historical outcomes for base rate calculation."""
    __tablename__ = "historical_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    season: Mapped[str] = mapped_column(String(20))  # "2025-26"
    market_type: Mapped[str] = mapped_column(String(20))  # "1x2", "ou", "btts"
    outcome: Mapped[str] = mapped_column(String(20))  # "home", "over", "yes"
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 2.5 for O/U

    # Stats
    total_matches: Mapped[int] = mapped_column(Integer, default=0)
    outcome_count: Mapped[int] = mapped_column(Integer, default=0)
    base_rate: Mapped[float] = mapped_column(Float, default=0.5)

    # Time-based rates (optional granularity)
    rate_home_matches: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rate_away_matches: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
