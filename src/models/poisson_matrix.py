"""Unified Poisson matrix for all market calculations."""
import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass


@dataclass
class PoissonResult:
    """Results from Poisson matrix calculation."""
    matrix: np.ndarray  # 2D probability matrix [home_goals][away_goals]
    home_xg: float
    away_xg: float

    # Derived probabilities
    p_1x2: dict  # {"home": float, "draw": float, "away": float}
    p_over_under: dict  # {"1.5": {"over": float, "under": float}, ...}
    p_btts: dict  # {"yes": float, "no": float}


class PoissonCalculator:
    """Calculate Poisson-based probabilities for football matches."""

    MAX_GOALS = 8  # Consider scorelines 0-0 to 7-7

    def calculate(self, home_xg: float, away_xg: float) -> PoissonResult:
        """Generate full Poisson matrix and derive all market probabilities.

        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team

        Returns:
            PoissonResult with matrix and all derived probabilities
        """
        # Clamp xG to reasonable range
        home_xg = max(0.1, min(home_xg, 5.0))
        away_xg = max(0.1, min(away_xg, 5.0))

        # Build probability matrix
        matrix = self._build_matrix(home_xg, away_xg)

        # Derive all markets from single matrix
        p_1x2 = self._calculate_1x2(matrix)
        p_over_under = self._calculate_over_under(matrix)
        p_btts = self._calculate_btts(matrix)

        return PoissonResult(
            matrix=matrix,
            home_xg=home_xg,
            away_xg=away_xg,
            p_1x2=p_1x2,
            p_over_under=p_over_under,
            p_btts=p_btts,
        )

    def _build_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        """Build probability matrix for all scorelines."""
        matrix = np.zeros((self.MAX_GOALS, self.MAX_GOALS))

        for h in range(self.MAX_GOALS):
            for a in range(self.MAX_GOALS):
                # Independent Poisson probabilities
                p_home = poisson.pmf(h, home_xg)
                p_away = poisson.pmf(a, away_xg)
                matrix[h, a] = p_home * p_away

        # Normalize (should sum to ~1, but ensure it)
        matrix = matrix / matrix.sum()

        return matrix

    def _calculate_1x2(self, matrix: np.ndarray) -> dict:
        """Calculate 1X2 (match result) probabilities."""
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for h in range(self.MAX_GOALS):
            for a in range(self.MAX_GOALS):
                prob = matrix[h, a]
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob

        return {
            "home": home_win,
            "draw": draw,
            "away": away_win,
        }

    def _calculate_over_under(self, matrix: np.ndarray) -> dict:
        """Calculate Over/Under probabilities for multiple lines."""
        lines = [1.5, 2.5, 3.5, 4.5]
        result = {}

        for line in lines:
            over_prob = 0.0

            for h in range(self.MAX_GOALS):
                for a in range(self.MAX_GOALS):
                    if h + a > line:  # Total goals > line = Over
                        over_prob += matrix[h, a]

            result[str(line)] = {
                "over": over_prob,
                "under": 1 - over_prob,
            }

        return result

    def _calculate_btts(self, matrix: np.ndarray) -> dict:
        """Calculate Both Teams To Score probabilities."""
        btts_yes = 0.0

        for h in range(1, self.MAX_GOALS):  # Home scores at least 1
            for a in range(1, self.MAX_GOALS):  # Away scores at least 1
                btts_yes += matrix[h, a]

        return {
            "yes": btts_yes,
            "no": 1 - btts_yes,
        }

    def get_most_likely_scores(
        self,
        matrix: np.ndarray,
        top_n: int = 5
    ) -> list[tuple[int, int, float]]:
        """Get most likely scorelines.

        Returns:
            List of (home_goals, away_goals, probability) tuples
        """
        scores = []
        for h in range(self.MAX_GOALS):
            for a in range(self.MAX_GOALS):
                scores.append((h, a, matrix[h, a]))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]


# Singleton instance
poisson_calc = PoissonCalculator()
