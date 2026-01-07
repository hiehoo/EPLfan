"""Backtesting engine for historical prediction validation."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from src.storage.database import db
from src.storage.models import Match, OddsSnapshot, Prediction

logger = logging.getLogger(__name__)


@dataclass
class BacktestParams:
    """Parameters for backtest run."""
    days: int = 180
    market: str = "all"  # 1x2, ou, btts, all
    min_edge: float = 0.0
    model_source: str = "hybrid"  # poisson, ml, hybrid


@dataclass
class BacktestResult:
    """Single prediction result."""
    match_id: int
    market_type: str
    predicted_prob: float
    market_prob: float
    edge: float
    actual_outcome: str
    is_correct: bool
    pnl: float  # +1 if win at odds, -1 if loss (flat stake)


@dataclass
class MarketStats:
    """Stats for a single market type."""
    n_predictions: int = 0
    n_correct: int = 0
    win_rate: float = 0.0
    roi_pct: float = 0.0
    avg_edge: float = 0.0


@dataclass
class BacktestReport:
    """Full backtest report."""
    period_start: datetime
    period_end: datetime
    n_predictions: int
    n_correct: int

    # ROI metrics
    win_rate: float
    roi_pct: float
    profit_factor: float

    # Statistical metrics
    brier_score: float
    calibration_error: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    max_win_streak: int
    max_lose_streak: int

    # By market
    by_market: dict = field(default_factory=dict)


class Backtester:
    """Core backtesting engine."""

    def run(self, params: BacktestParams) -> BacktestReport:
        """Run backtest with given parameters."""
        results = self._load_predictions(params.days, params.market, params.min_edge)

        if not results:
            logger.warning("No predictions found for backtest")
            now = datetime.now(timezone.utc)
            return BacktestReport(
                period_start=now - timedelta(days=params.days),
                period_end=now,
                n_predictions=0,
                n_correct=0,
                win_rate=0.0,
                roi_pct=0.0,
                profit_factor=0.0,
                brier_score=0.0,
                calibration_error=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                max_win_streak=0,
                max_lose_streak=0,
            )

        return self._calculate_metrics(results, params.days)

    def _load_predictions(
        self,
        days: int,
        market: str,
        min_edge: float,
    ) -> list[BacktestResult]:
        """Query completed predictions from DB."""
        results = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with db.session() as session:
            # Get completed matches with predictions
            matches = (
                session.query(Match)
                .filter(
                    Match.is_completed == True,
                    Match.kickoff >= cutoff,
                )
                .all()
            )

            for match in matches:
                # Get latest odds snapshot
                odds = (
                    session.query(OddsSnapshot)
                    .filter_by(match_id=match.id)
                    .order_by(OddsSnapshot.snapshot_time.desc())
                    .first()
                )

                if not odds:
                    continue

                # Calculate outcomes
                total_goals = (match.home_goals or 0) + (match.away_goals or 0)
                btts = (match.home_goals or 0) > 0 and (match.away_goals or 0) > 0

                # Generate backtest results for each market
                if market in ("all", "ou"):
                    ou_result = self._create_ou_result(match, odds, total_goals, min_edge)
                    if ou_result:
                        results.append(ou_result)

                if market in ("all", "btts"):
                    btts_result = self._create_btts_result(match, odds, btts, min_edge)
                    if btts_result:
                        results.append(btts_result)

                if market in ("all", "1x2"):
                    result_1x2 = self._create_1x2_result(match, odds, min_edge)
                    if result_1x2:
                        results.append(result_1x2)

        logger.info(f"Loaded {len(results)} predictions for backtest")
        return results

    def _create_ou_result(
        self,
        match: Match,
        odds: OddsSnapshot,
        total_goals: int,
        min_edge: float,
    ) -> Optional[BacktestResult]:
        """Create O/U backtest result."""
        if not odds.bf_over_2_5_odds or not odds.bf_under_2_5_odds:
            return None

        # Market implied probability
        market_over = 1.0 / odds.bf_over_2_5_odds
        market_under = 1.0 / odds.bf_under_2_5_odds

        # Normalize to remove vig
        total = market_over + market_under
        market_over /= total
        market_under /= total

        # Simple model: use historical base rate (52% over in EPL)
        model_over = 0.52

        # Calculate edge
        edge = model_over - market_over

        if abs(edge) < min_edge:
            return None

        # Determine prediction and outcome
        predicted_over = edge > 0
        actual_over = total_goals > 2.5

        # PnL: +1 if correct at market odds, -1 if wrong
        if predicted_over:
            is_correct = actual_over
            pnl = (odds.bf_over_2_5_odds - 1) if is_correct else -1.0
        else:
            is_correct = not actual_over
            pnl = (odds.bf_under_2_5_odds - 1) if is_correct else -1.0

        return BacktestResult(
            match_id=match.id,
            market_type="ou",
            predicted_prob=model_over,
            market_prob=market_over,
            edge=edge,
            actual_outcome="over" if actual_over else "under",
            is_correct=is_correct,
            pnl=pnl,
        )

    def _create_btts_result(
        self,
        match: Match,
        odds: OddsSnapshot,
        btts_actual: bool,
        min_edge: float,
    ) -> Optional[BacktestResult]:
        """Create BTTS backtest result."""
        if not odds.bf_btts_yes_odds or not odds.bf_btts_no_odds:
            return None

        # Market implied probability
        market_yes = 1.0 / odds.bf_btts_yes_odds
        market_no = 1.0 / odds.bf_btts_no_odds

        # Normalize
        total = market_yes + market_no
        market_yes /= total
        market_no /= total

        # Simple model: use historical base rate (48% BTTS in EPL)
        model_yes = 0.48

        edge = model_yes - market_yes

        if abs(edge) < min_edge:
            return None

        predicted_yes = edge > 0

        if predicted_yes:
            is_correct = btts_actual
            pnl = (odds.bf_btts_yes_odds - 1) if is_correct else -1.0
        else:
            is_correct = not btts_actual
            pnl = (odds.bf_btts_no_odds - 1) if is_correct else -1.0

        return BacktestResult(
            match_id=match.id,
            market_type="btts",
            predicted_prob=model_yes,
            market_prob=market_yes,
            edge=edge,
            actual_outcome="yes" if btts_actual else "no",
            is_correct=is_correct,
            pnl=pnl,
        )

    def _create_1x2_result(
        self,
        match: Match,
        odds: OddsSnapshot,
        min_edge: float,
    ) -> Optional[BacktestResult]:
        """Create 1X2 backtest result."""
        if not odds.bf_home_odds or not odds.bf_draw_odds or not odds.bf_away_odds:
            return None

        # Market implied
        market_home = 1.0 / odds.bf_home_odds
        market_draw = 1.0 / odds.bf_draw_odds
        market_away = 1.0 / odds.bf_away_odds

        total = market_home + market_draw + market_away
        market_home /= total
        market_draw /= total
        market_away /= total

        # Determine actual outcome
        home_goals = match.home_goals or 0
        away_goals = match.away_goals or 0

        if home_goals > away_goals:
            actual = "home"
        elif away_goals > home_goals:
            actual = "away"
        else:
            actual = "draw"

        # Simple model: slight home advantage (EPL home win ~45%)
        model_home = 0.45
        model_draw = 0.27
        model_away = 0.28

        # Find best edge
        edges = {
            "home": model_home - market_home,
            "draw": model_draw - market_draw,
            "away": model_away - market_away,
        }

        best_selection = max(edges, key=edges.get)
        best_edge = edges[best_selection]

        if best_edge < min_edge:
            return None

        is_correct = best_selection == actual

        # PnL at market odds
        odds_map = {
            "home": odds.bf_home_odds,
            "draw": odds.bf_draw_odds,
            "away": odds.bf_away_odds,
        }
        pnl = (odds_map[best_selection] - 1) if is_correct else -1.0

        model_prob = {"home": model_home, "draw": model_draw, "away": model_away}[best_selection]
        market_prob = {"home": market_home, "draw": market_draw, "away": market_away}[best_selection]

        return BacktestResult(
            match_id=match.id,
            market_type="1x2",
            predicted_prob=model_prob,
            market_prob=market_prob,
            edge=best_edge,
            actual_outcome=actual,
            is_correct=is_correct,
            pnl=pnl,
        )

    def _calculate_metrics(
        self,
        results: list[BacktestResult],
        days: int,
    ) -> BacktestReport:
        """Calculate all metrics from results."""
        n = len(results)
        n_correct = sum(1 for r in results if r.is_correct)

        # Basic metrics
        win_rate = n_correct / n if n else 0
        total_pnl = sum(r.pnl for r in results)
        roi_pct = (total_pnl / n * 100) if n else 0

        # Profit factor
        wins = sum(r.pnl for r in results if r.pnl > 0)
        losses = abs(sum(r.pnl for r in results if r.pnl < 0))
        profit_factor = wins / losses if losses > 0 else float("inf") if wins > 0 else 0

        # Brier score
        brier = self._calculate_brier_score(results)

        # Calibration error
        calibration = self._calculate_calibration_error(results)

        # Risk metrics
        pnl_series = [r.pnl for r in results]
        sharpe = self._calculate_sharpe(pnl_series)
        drawdown = self._calculate_max_drawdown(pnl_series)
        win_streak, lose_streak = self._calculate_streaks(results)

        # By market breakdown
        by_market = {}
        for market in ["1x2", "ou", "btts"]:
            market_results = [r for r in results if r.market_type == market]
            if market_results:
                m_correct = sum(1 for r in market_results if r.is_correct)
                m_pnl = sum(r.pnl for r in market_results)
                by_market[market] = MarketStats(
                    n_predictions=len(market_results),
                    n_correct=m_correct,
                    win_rate=m_correct / len(market_results),
                    roi_pct=(m_pnl / len(market_results) * 100),
                    avg_edge=sum(r.edge for r in market_results) / len(market_results),
                )

        now = datetime.now(timezone.utc)
        return BacktestReport(
            period_start=now - timedelta(days=days),
            period_end=now,
            n_predictions=n,
            n_correct=n_correct,
            win_rate=win_rate,
            roi_pct=roi_pct,
            profit_factor=profit_factor,
            brier_score=brier,
            calibration_error=calibration,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            max_win_streak=win_streak,
            max_lose_streak=lose_streak,
            by_market=by_market,
        )

    def _calculate_brier_score(self, results: list[BacktestResult]) -> float:
        """Brier = mean((prob - outcome)^2)"""
        if not results:
            return 0.0

        scores = []
        for r in results:
            outcome = 1.0 if r.is_correct else 0.0
            scores.append((r.predicted_prob - outcome) ** 2)

        return float(np.mean(scores))

    def _calculate_calibration_error(self, results: list[BacktestResult]) -> float:
        """Calculate expected calibration error."""
        if not results:
            return 0.0

        # Bin predictions by probability
        bins = np.linspace(0, 1, 11)
        bin_errors = []

        for i in range(len(bins) - 1):
            bin_results = [
                r for r in results
                if bins[i] <= r.predicted_prob < bins[i + 1]
            ]
            if bin_results:
                avg_prob = np.mean([r.predicted_prob for r in bin_results])
                actual_rate = np.mean([1.0 if r.is_correct else 0.0 for r in bin_results])
                bin_errors.append(abs(avg_prob - actual_rate))

        return float(np.mean(bin_errors)) if bin_errors else 0.0

    def _calculate_sharpe(self, pnl_series: list[float]) -> float:
        """Sharpe = mean(returns) / std(returns)"""
        if len(pnl_series) < 2:
            return 0.0

        mean_pnl = np.mean(pnl_series)
        std_pnl = np.std(pnl_series)

        if std_pnl == 0:
            return 0.0

        return float(mean_pnl / std_pnl)

    def _calculate_max_drawdown(self, pnl_series: list[float]) -> float:
        """Calculate maximum drawdown from peak."""
        if not pnl_series:
            return 0.0

        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calculate_streaks(self, results: list[BacktestResult]) -> tuple[int, int]:
        """Calculate max win and lose streaks."""
        if not results:
            return 0, 0

        max_win = 0
        max_lose = 0
        current_win = 0
        current_lose = 0

        for r in results:
            if r.is_correct:
                current_win += 1
                current_lose = 0
                max_win = max(max_win, current_win)
            else:
                current_lose += 1
                current_win = 0
                max_lose = max(max_lose, current_lose)

        return max_win, max_lose

    def to_json(self, report: BacktestReport) -> str:
        """Export report as JSON."""
        data = {
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "n_predictions": report.n_predictions,
            "n_correct": report.n_correct,
            "win_rate": round(report.win_rate, 4),
            "roi_pct": round(report.roi_pct, 2),
            "profit_factor": round(report.profit_factor, 2),
            "brier_score": round(report.brier_score, 4),
            "calibration_error": round(report.calibration_error, 4),
            "sharpe_ratio": round(report.sharpe_ratio, 2),
            "max_drawdown": round(report.max_drawdown, 2),
            "max_win_streak": report.max_win_streak,
            "max_lose_streak": report.max_lose_streak,
            "by_market": {
                k: {
                    "n_predictions": v.n_predictions,
                    "n_correct": v.n_correct,
                    "win_rate": round(v.win_rate, 4),
                    "roi_pct": round(v.roi_pct, 2),
                    "avg_edge": round(v.avg_edge, 4),
                }
                for k, v in report.by_market.items()
            },
        }
        return json.dumps(data, indent=2)


# Singleton instance
backtester = Backtester()
