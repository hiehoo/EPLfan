"""Apply market-specific weights to calculate fair probabilities."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class WeightedProbabilities:
    """Fair probabilities after applying weights."""
    preset_name: str
    hours_to_kickoff: float

    # 1X2 fair probabilities
    fair_1x2: dict  # {"home": float, "draw": float, "away": float}

    # O/U fair probabilities
    fair_ou: dict  # {"2.5": {"over": float, "under": float}}

    # BTTS fair probabilities
    fair_btts: dict  # {"yes": float, "no": float}

    # Component breakdown (for UI)
    breakdown_1x2: dict  # {"betfair": {...}, "xg": {...}, "elo": {...}, "form": {...}}
    breakdown_ou: dict
    breakdown_btts: dict


class WeightEngine:
    """Apply weights to calculate fair probabilities."""

    def __init__(self, weights_dir: Path = None):
        self.weights_dir = weights_dir or Path(__file__).parent.parent / "config" / "weights"
        self._load_weight_configs()

    def _load_weight_configs(self) -> None:
        """Load weight configurations from YAML files."""
        self.weights = {
            "1x2": self._load_yaml("1x2_weights.yaml"),
            "ou": self._load_yaml("ou_weights.yaml"),
            "btts": self._load_yaml("btts_weights.yaml"),
        }

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML weight file."""
        filepath = self.weights_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                return yaml.safe_load(f)
        logger.warning(f"Weight file not found: {filepath}")
        return {}

    def get_preset_for_timing(
        self,
        market_type: str,
        hours_to_kickoff: float
    ) -> str:
        """Get recommended preset based on hours to kickoff."""
        thresholds = self.weights.get(market_type, {}).get("time_thresholds", {})

        if hours_to_kickoff > thresholds.get("analytics_first", 96):
            return "analytics_first"
        elif hours_to_kickoff > thresholds.get("balanced", 24):
            return "balanced"
        else:
            return "market_trust"

    def calculate_fair_probabilities(
        self,
        # Probability inputs
        p_betfair: dict,
        p_xg: dict,
        p_elo: dict,
        p_form: dict,  # Can be same as p_xg for simplicity
        # Match context
        hours_to_kickoff: float,
        market_type: str = "1x2",
        preset_override: Optional[str] = None,
    ) -> tuple[dict, str, dict]:
        """Calculate weighted fair probabilities.

        Args:
            p_betfair: Betfair implied probabilities
            p_xg: Poisson xG-derived probabilities
            p_elo: ELO-derived probabilities
            p_form: Form-based probabilities (recent matches)
            hours_to_kickoff: Hours until match starts
            market_type: "1x2", "ou", or "btts"
            preset_override: Force specific preset instead of time-based

        Returns:
            (fair_probabilities, preset_used, breakdown)
        """
        # Select preset
        preset = preset_override or self.get_preset_for_timing(market_type, hours_to_kickoff)
        weights = self.weights.get(market_type, {}).get(preset, {})

        if not weights:
            logger.warning(f"No weights for {market_type}/{preset}, using equal")
            weights = {"betfair": 0.25, "xg": 0.25, "elo": 0.25, "form": 0.25}

        # Get weight values
        w_betfair = weights.get("betfair", 0.4)
        w_xg = weights.get("xg", 0.3)
        w_elo = weights.get("elo", weights.get("xga", 0.2))  # xga for BTTS
        w_form = weights.get("form", 0.1)

        # Calculate weighted probabilities for each outcome
        fair_probs = {}
        breakdown = {
            "betfair": {},
            "xg": {},
            "elo": {},
            "form": {},
            "weights": weights,
        }

        for outcome in p_betfair.keys():
            weighted = (
                w_betfair * p_betfair.get(outcome, 0) +
                w_xg * p_xg.get(outcome, 0) +
                w_elo * p_elo.get(outcome, 0) +
                w_form * p_form.get(outcome, 0)
            )
            fair_probs[outcome] = weighted

            # Store breakdown
            breakdown["betfair"][outcome] = p_betfair.get(outcome, 0)
            breakdown["xg"][outcome] = p_xg.get(outcome, 0)
            breakdown["elo"][outcome] = p_elo.get(outcome, 0)
            breakdown["form"][outcome] = p_form.get(outcome, 0)

        # Normalize to ensure sum = 1
        total = sum(fair_probs.values())
        if total > 0:
            fair_probs = {k: v / total for k, v in fair_probs.items()}

        return fair_probs, preset, breakdown

    def calculate_all_markets(
        self,
        match_probs,  # MatchProbabilities object
        hours_to_kickoff: float,
        preset_1x2: Optional[str] = None,
        preset_ou: Optional[str] = None,
        preset_btts: Optional[str] = None,
    ) -> WeightedProbabilities:
        """Calculate fair probabilities for all markets."""

        # 1X2
        fair_1x2, preset_1x2_used, breakdown_1x2 = self.calculate_fair_probabilities(
            p_betfair=match_probs.betfair_1x2,
            p_xg=match_probs.poisson_result.p_1x2,
            p_elo=match_probs.elo_1x2,
            p_form=match_probs.poisson_result.p_1x2,  # Use xG as form proxy
            hours_to_kickoff=hours_to_kickoff,
            market_type="1x2",
            preset_override=preset_1x2,
        )

        # O/U 2.5 (main line)
        p_xg_ou = match_probs.poisson_result.p_over_under.get("2.5", {"over": 0.5, "under": 0.5})
        p_betfair_ou = match_probs.betfair_ou.get("2.5", {"over": 0.5, "under": 0.5})

        fair_ou_2_5, preset_ou_used, breakdown_ou = self.calculate_fair_probabilities(
            p_betfair=p_betfair_ou,
            p_xg=p_xg_ou,
            p_elo={"over": 0.5, "under": 0.5},  # ELO not relevant for O/U
            p_form=p_xg_ou,
            hours_to_kickoff=hours_to_kickoff,
            market_type="ou",
            preset_override=preset_ou,
        )

        fair_ou = {"2.5": fair_ou_2_5}

        # BTTS
        p_xg_btts = match_probs.poisson_result.p_btts
        p_betfair_btts = match_probs.betfair_btts or {"yes": 0.5, "no": 0.5}

        fair_btts, preset_btts_used, breakdown_btts = self.calculate_fair_probabilities(
            p_betfair=p_betfair_btts,
            p_xg=p_xg_btts,
            p_elo=p_xg_btts,  # Use xGA-adjusted for BTTS
            p_form=p_xg_btts,
            hours_to_kickoff=hours_to_kickoff,
            market_type="btts",
            preset_override=preset_btts,
        )

        return WeightedProbabilities(
            preset_name=preset_1x2_used,  # Main preset
            hours_to_kickoff=hours_to_kickoff,
            fair_1x2=fair_1x2,
            fair_ou=fair_ou,
            fair_btts=fair_btts,
            breakdown_1x2=breakdown_1x2,
            breakdown_ou=breakdown_ou,
            breakdown_btts=breakdown_btts,
        )


# Singleton instance
weight_engine = WeightEngine()
