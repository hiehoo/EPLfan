"""Track historical base rates for true edge calculation."""
import logging
from dataclasses import dataclass
from typing import Optional

from src.storage.database import db
from src.storage.models import Match

logger = logging.getLogger(__name__)


@dataclass
class BaseRates:
    """Base rates for all markets."""
    # 1X2
    home_rate: float = 0.46  # EPL historical avg
    draw_rate: float = 0.25
    away_rate: float = 0.29

    # O/U 2.5
    over_2_5_rate: float = 0.52
    under_2_5_rate: float = 0.48

    # BTTS
    btts_yes_rate: float = 0.53
    btts_no_rate: float = 0.47


class BaseRateTracker:
    """Track and update historical base rates."""

    # Default EPL base rates (2015-2025 historical data)
    DEFAULTS = BaseRates()

    # Elite teams with distinct patterns
    ELITE_TEAMS = [
        "Manchester City",
        "Liverpool",
        "Arsenal",
        "Chelsea",
    ]
    ELITE_ADJUSTMENT = 1.15  # 15% confidence boost

    def __init__(self):
        self._cache: Optional[BaseRates] = None
        self._elite_cache: dict[str, float] = {}

    def get_base_rate(
        self,
        market_type: str,
        outcome: str,
        line: Optional[float] = None
    ) -> float:
        """Get historical base rate for an outcome.

        Args:
            market_type: "1x2", "ou", "btts"
            outcome: "home", "over", "yes", etc.
            line: O/U line (e.g., 2.5)

        Returns:
            Historical probability of this outcome
        """
        rates = self._get_cached_rates()

        if market_type == "1x2":
            return {
                "home": rates.home_rate,
                "draw": rates.draw_rate,
                "away": rates.away_rate,
            }.get(outcome, 0.33)

        elif market_type == "ou":
            if line == 2.5 or line is None:
                return {
                    "over": rates.over_2_5_rate,
                    "under": rates.under_2_5_rate,
                }.get(outcome, 0.5)
            # Other lines - use default 0.5
            return 0.5

        elif market_type == "btts":
            return {
                "yes": rates.btts_yes_rate,
                "no": rates.btts_no_rate,
            }.get(outcome, 0.5)

        return 0.5

    def get_elite_adjustment(self, team: str) -> float:
        """Get adjustment factor for elite teams."""
        if team in self.ELITE_TEAMS:
            return self.ELITE_ADJUSTMENT
        return 1.0

    def is_elite_team(self, team: str) -> bool:
        """Check if team is in elite category."""
        return team in self.ELITE_TEAMS

    def update_from_completed_matches(self) -> None:
        """Update base rates from completed matches in database."""
        try:
            with db.session() as session:
                # Query completed matches
                matches = session.query(Match).filter(
                    Match.is_completed == True  # noqa: E712
                ).all()

                if len(matches) < 50:
                    logger.warning(f"Only {len(matches)} completed matches, using defaults")
                    return

                # Calculate rates
                total = len(matches)
                home_wins = sum(1 for m in matches if m.home_goals > m.away_goals)
                draws = sum(1 for m in matches if m.home_goals == m.away_goals)
                over_2_5 = sum(1 for m in matches if (m.home_goals + m.away_goals) > 2.5)
                btts_yes = sum(1 for m in matches if m.home_goals > 0 and m.away_goals > 0)

                self._cache = BaseRates(
                    home_rate=home_wins / total,
                    draw_rate=draws / total,
                    away_rate=(total - home_wins - draws) / total,
                    over_2_5_rate=over_2_5 / total,
                    under_2_5_rate=(total - over_2_5) / total,
                    btts_yes_rate=btts_yes / total,
                    btts_no_rate=(total - btts_yes) / total,
                )

                logger.info(f"Updated base rates from {total} matches")
        except Exception as e:
            logger.warning(f"Failed to update base rates from DB: {e}, using defaults")

    def _get_cached_rates(self) -> BaseRates:
        """Get cached rates or load from DB."""
        if self._cache is None:
            self.update_from_completed_matches()
        return self._cache or self.DEFAULTS


# Singleton
base_rate_tracker = BaseRateTracker()
