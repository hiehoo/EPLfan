"""A/B testing framework for Poisson vs ML comparison."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path

from src.storage.database import db
from src.storage.models import Match

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of A/B test for a single prediction."""
    match_id: str
    market_type: str
    outcome: str

    # Predictions
    poisson_prob: float
    ml_prob: Optional[float]
    hybrid_prob: float
    market_prob: float

    # Actual result
    actual_outcome: Optional[str] = None
    is_correct_poisson: Optional[bool] = None
    is_correct_ml: Optional[bool] = None
    is_correct_hybrid: Optional[bool] = None

    # Edge metrics
    poisson_edge: float = 0.0
    ml_edge: Optional[float] = None
    hybrid_edge: float = 0.0


@dataclass
class ABTestSummary:
    """Summary statistics for A/B test."""
    period_start: datetime
    period_end: datetime
    n_predictions: int

    # Win rates
    poisson_win_rate: float = 0.0
    ml_win_rate: float = 0.0
    hybrid_win_rate: float = 0.0

    # Edge metrics
    poisson_avg_edge: float = 0.0
    ml_avg_edge: float = 0.0
    hybrid_avg_edge: float = 0.0

    # Agreement
    agreement_rate: float = 0.0  # How often Poisson and ML agree

    # By market
    by_market: dict = field(default_factory=dict)


class ABTestTracker:
    """Track A/B test results."""

    RESULTS_DIR = Path("data/ab_testing")

    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.results: list[ABTestResult] = []

    def log_prediction(
        self,
        match_id: str,
        market_type: str,
        outcome: str,
        poisson_prob: float,
        ml_prob: Optional[float],
        hybrid_prob: float,
        market_prob: float,
    ):
        """Log a prediction for later evaluation."""
        result = ABTestResult(
            match_id=match_id,
            market_type=market_type,
            outcome=outcome,
            poisson_prob=poisson_prob,
            ml_prob=ml_prob,
            hybrid_prob=hybrid_prob,
            market_prob=market_prob,
            poisson_edge=poisson_prob - market_prob,
            ml_edge=(ml_prob - market_prob) if ml_prob else None,
            hybrid_edge=hybrid_prob - market_prob,
        )
        self.results.append(result)

    def evaluate_completed_matches(self) -> int:
        """Evaluate predictions against actual results."""
        evaluated = 0

        with db.session() as session:
            for result in self.results:
                if result.actual_outcome is not None:
                    continue  # Already evaluated

                match = session.query(Match).filter(
                    Match.id == int(result.match_id),
                    Match.is_completed == True,
                ).first()

                if not match:
                    continue

                # Determine actual outcome
                if result.market_type == "ou":
                    total = match.home_goals + match.away_goals
                    actual = "over" if total > 2.5 else "under"
                elif result.market_type == "btts":
                    actual = "yes" if (match.home_goals > 0 and match.away_goals > 0) else "no"
                else:
                    continue

                result.actual_outcome = actual

                # Check correctness
                predicted_poisson = "over" if result.poisson_prob > 0.5 else "under"
                predicted_ml = "over" if (result.ml_prob or 0.5) > 0.5 else "under"
                predicted_hybrid = "over" if result.hybrid_prob > 0.5 else "under"

                if result.market_type == "btts":
                    predicted_poisson = "yes" if result.poisson_prob > 0.5 else "no"
                    predicted_ml = "yes" if (result.ml_prob or 0.5) > 0.5 else "no"
                    predicted_hybrid = "yes" if result.hybrid_prob > 0.5 else "no"

                result.is_correct_poisson = predicted_poisson == actual
                result.is_correct_ml = predicted_ml == actual if result.ml_prob else None
                result.is_correct_hybrid = predicted_hybrid == actual

                evaluated += 1

        return evaluated

    def get_summary(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ABTestSummary:
        """Get summary statistics."""
        results = self.results
        if not results:
            return ABTestSummary(
                period_start=period_start or datetime.now(timezone.utc),
                period_end=period_end or datetime.now(timezone.utc),
                n_predictions=0,
            )

        # Evaluate any pending
        self.evaluate_completed_matches()

        # Filter to evaluated only
        evaluated = [r for r in results if r.actual_outcome is not None]

        if not evaluated:
            return ABTestSummary(
                period_start=period_start or datetime.now(timezone.utc),
                period_end=period_end or datetime.now(timezone.utc),
                n_predictions=len(results),
            )

        # Calculate metrics
        poisson_correct = sum(1 for r in evaluated if r.is_correct_poisson)
        ml_results = [r for r in evaluated if r.is_correct_ml is not None]
        ml_correct = sum(1 for r in ml_results if r.is_correct_ml)
        hybrid_correct = sum(1 for r in evaluated if r.is_correct_hybrid)

        # Agreement rate
        agreements = sum(
            1 for r in evaluated
            if r.ml_prob and (r.poisson_prob > 0.5) == (r.ml_prob > 0.5)
        )
        ml_count = len(ml_results)

        return ABTestSummary(
            period_start=period_start or datetime.now(timezone.utc),
            period_end=period_end or datetime.now(timezone.utc),
            n_predictions=len(evaluated),
            poisson_win_rate=poisson_correct / len(evaluated) if evaluated else 0,
            ml_win_rate=ml_correct / len(ml_results) if ml_results else 0,
            hybrid_win_rate=hybrid_correct / len(evaluated) if evaluated else 0,
            poisson_avg_edge=sum(r.poisson_edge for r in evaluated) / len(evaluated),
            ml_avg_edge=sum(r.ml_edge for r in ml_results if r.ml_edge) / len(ml_results) if ml_results else 0,
            hybrid_avg_edge=sum(r.hybrid_edge for r in evaluated) / len(evaluated),
            agreement_rate=agreements / ml_count if ml_count else 0,
        )

    def save_results(self):
        """Save results to disk."""
        filepath = self.RESULTS_DIR / f"ab_results_{datetime.now().strftime('%Y%m%d')}.json"
        data = [
            {
                "match_id": r.match_id,
                "market_type": r.market_type,
                "outcome": r.outcome,
                "poisson_prob": r.poisson_prob,
                "ml_prob": r.ml_prob,
                "hybrid_prob": r.hybrid_prob,
                "actual_outcome": r.actual_outcome,
                "is_correct_poisson": r.is_correct_poisson,
                "is_correct_ml": r.is_correct_ml,
                "is_correct_hybrid": r.is_correct_hybrid,
            }
            for r in self.results
        ]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_results(self, date: Optional[str] = None):
        """Load results from disk."""
        date = date or datetime.now().strftime("%Y%m%d")
        filepath = self.RESULTS_DIR / f"ab_results_{date}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                self.results = [
                    ABTestResult(
                        match_id=r["match_id"],
                        market_type=r["market_type"],
                        outcome=r["outcome"],
                        poisson_prob=r["poisson_prob"],
                        ml_prob=r["ml_prob"],
                        hybrid_prob=r["hybrid_prob"],
                        market_prob=0.5,  # Default, not saved
                        actual_outcome=r.get("actual_outcome"),
                        is_correct_poisson=r.get("is_correct_poisson"),
                        is_correct_ml=r.get("is_correct_ml"),
                        is_correct_hybrid=r.get("is_correct_hybrid"),
                    )
                    for r in data
                ]


# Singleton
ab_tracker = ABTestTracker()
