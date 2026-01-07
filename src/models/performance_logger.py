"""Log model performance for monitoring."""
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PerformanceLogger:
    """Log performance metrics for monitoring."""

    LOG_DIR = Path("data/performance")

    def __init__(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        match_id: str,
        market_type: str,
        poisson_prob: float,
        ml_prob: float,
        hybrid_prob: float,
        market_prob: float,
        true_edge: float,
        is_actionable: bool,
    ):
        """Log a prediction for performance tracking."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "match_id": match_id,
            "market_type": market_type,
            "poisson_prob": poisson_prob,
            "ml_prob": ml_prob,
            "hybrid_prob": hybrid_prob,
            "market_prob": market_prob,
            "true_edge": true_edge,
            "is_actionable": is_actionable,
        }

        # Append to daily log
        today = datetime.utcnow().strftime("%Y%m%d")
        filepath = self.LOG_DIR / f"predictions_{today}.jsonl"

        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_daily_stats(self, date: str = None) -> dict:
        """Get statistics for a day."""
        date = date or datetime.utcnow().strftime("%Y%m%d")
        filepath = self.LOG_DIR / f"predictions_{date}.jsonl"

        if not filepath.exists():
            return {"n_predictions": 0}

        entries = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if not entries:
            return {"n_predictions": 0}

        actionable = [e for e in entries if e["is_actionable"]]

        return {
            "n_predictions": len(entries),
            "n_actionable": len(actionable),
            "avg_poisson_prob": sum(e["poisson_prob"] for e in entries) / len(entries),
            "avg_ml_prob": sum(e["ml_prob"] for e in entries) / len(entries),
            "avg_true_edge": sum(e["true_edge"] for e in entries) / len(entries),
            "by_market": self._group_by_market(entries),
        }

    def _group_by_market(self, entries: list) -> dict:
        """Group stats by market type."""
        by_market = {}
        for e in entries:
            mt = e["market_type"]
            if mt not in by_market:
                by_market[mt] = {"count": 0, "actionable": 0, "total_edge": 0}
            by_market[mt]["count"] += 1
            by_market[mt]["actionable"] += 1 if e["is_actionable"] else 0
            by_market[mt]["total_edge"] += e["true_edge"]

        for mt, stats in by_market.items():
            if stats["count"] > 0:
                stats["avg_edge"] = stats["total_edge"] / stats["count"]

        return by_market

    def get_weekly_trend(self) -> list[dict]:
        """Get stats for last 7 days."""
        from datetime import timedelta

        today = datetime.utcnow()
        trend = []

        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y%m%d")
            stats = self.get_daily_stats(date)
            stats["date"] = date
            trend.append(stats)

        return list(reversed(trend))


# Singleton
performance_logger = PerformanceLogger()
