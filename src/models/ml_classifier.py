"""ML classification ensemble for O/U predictions."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Optional imports (graceful degradation)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from ML classification."""
    market_type: str  # "ou", "btts"
    outcome: str  # "over", "under", "yes", "no"
    probability: float
    confidence: str  # "high", "medium", "low"
    model_used: str  # "ensemble", "lgb", "xgb", "rf"

    # Ensemble breakdown
    lgb_prob: Optional[float] = None
    xgb_prob: Optional[float] = None
    rf_prob: Optional[float] = None


class MLClassifier:
    """Ensemble classifier for O/U and BTTS predictions."""

    MODEL_DIR = Path("data/models")

    def __init__(self, target: str = "ou_2_5"):
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models = {}
        self._init_models()

        # Track if trained
        self.is_trained = False
        self.trained_models = {}
        self.target = target

    def _init_models(self):
        """Initialize ensemble models."""
        # LightGBM (if available)
        if HAS_LIGHTGBM:
            self.models["lgb"] = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )

        # XGBoost (if available)
        if HAS_XGBOOST:
            self.models["xgb"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            )

        # RandomForest (always available)
        self.models["rf"] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )

        logger.info(f"Initialized models: {list(self.models.keys())}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: str = "ou_2_5",  # "ou_2_5", "btts"
        calibrate: bool = True,
    ) -> dict:
        """Train ensemble models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - binary 0/1
            target: Target market ("ou_2_5", "btts")
            calibrate: Apply probability calibration

        Returns:
            Training metrics dict
        """
        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples, need more data")
            return {"error": "insufficient_data"}

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        metrics = {"target": target, "n_train": len(X_train), "n_val": len(X_val)}

        # Train each model
        trained_models = {}
        for name, model in self.models.items():
            try:
                if calibrate:
                    # Wrap with calibration
                    calibrated = CalibratedClassifierCV(model, cv=3)
                    calibrated.fit(X_train, y_train)
                    trained_models[name] = calibrated
                else:
                    model.fit(X_train, y_train)
                    trained_models[name] = model

                # Evaluate
                val_preds = trained_models[name].predict_proba(X_val)[:, 1]
                val_accuracy = np.mean((val_preds > 0.5) == y_val)
                metrics[f"{name}_accuracy"] = float(val_accuracy)

                logger.info(f"{name} accuracy: {val_accuracy:.3f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                metrics[f"{name}_error"] = str(e)

        # Store trained models
        self.trained_models = trained_models
        self.target = target
        self.is_trained = True

        # Save models
        self._save_models(target)

        return metrics

    def predict_proba(self, features: np.ndarray) -> ClassificationResult:
        """Get probability prediction from ensemble.

        Args:
            features: Feature vector (14,)

        Returns:
            ClassificationResult with probabilities
        """
        if not self.is_trained:
            # Try to load saved models
            if not self._load_models():
                logger.warning("Models not trained, returning default")
                return ClassificationResult(
                    market_type="ou",
                    outcome="over",
                    probability=0.5,
                    confidence="low",
                    model_used="none",
                )

        # Validate input type
        if not isinstance(features, np.ndarray):
            logger.warning(f"Invalid features type: {type(features)}, expected ndarray")
            return ClassificationResult(
                market_type="ou",
                outcome="over",
                probability=0.5,
                confidence="low",
                model_used="none",
            )

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get predictions from each model
        probs = {}
        for name, model in self.trained_models.items():
            try:
                prob = model.predict_proba(features)[0, 1]  # P(positive class)
                probs[name] = prob
            except Exception as e:
                logger.debug(f"Error predicting with {name}: {e}")

        if not probs:
            return ClassificationResult(
                market_type="ou",
                outcome="over",
                probability=0.5,
                confidence="low",
                model_used="none",
            )

        # Ensemble average
        ensemble_prob = np.mean(list(probs.values()))

        # Determine confidence based on model agreement
        std_dev = np.std(list(probs.values()))
        if std_dev < 0.05:
            confidence = "high"
        elif std_dev < 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine outcome
        outcome = "over" if ensemble_prob > 0.5 else "under"
        if self.target == "btts":
            outcome = "yes" if ensemble_prob > 0.5 else "no"

        return ClassificationResult(
            market_type="ou" if "ou" in self.target else "btts",
            outcome=outcome,
            probability=ensemble_prob,
            confidence=confidence,
            model_used="ensemble",
            lgb_prob=probs.get("lgb"),
            xgb_prob=probs.get("xgb"),
            rf_prob=probs.get("rf"),
        )

    def _save_models(self, target: str):
        """Save trained models to disk."""
        filepath = self.MODEL_DIR / f"classifier_{target}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "models": self.trained_models,
                "target": target,
            }, f)
        logger.info(f"Saved models to {filepath}")

    def _load_models(self, target: str = None) -> bool:
        """Load trained models from disk."""
        target = target or self.target
        filepath = self.MODEL_DIR / f"classifier_{target}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.trained_models = data["models"]
                self.target = data["target"]
                self.is_trained = True
                logger.info(f"Loaded models from {filepath}")
                return True
        return False

    def get_feature_importance(self) -> dict:
        """Get feature importance from RF model."""
        if "rf" not in self.trained_models:
            return {}

        model = self.trained_models["rf"]
        if hasattr(model, "feature_importances_"):
            return dict(zip(
                self._get_feature_names(),
                model.feature_importances_
            ))

        # For calibrated classifier
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "feature_importances_"):
                return dict(zip(
                    self._get_feature_names(),
                    base.feature_importances_
                ))

        return {}

    def _get_feature_names(self) -> list[str]:
        """Get feature names (match MatchFeatures.feature_names)."""
        from src.models.feature_builder import MatchFeatures
        return MatchFeatures(
            match_id="",
            home_team="",
            away_team="",
        ).feature_names


# Singletons for each market
ml_classifier_ou = MLClassifier(target="ou_2_5")
ml_classifier_btts = MLClassifier(target="btts")
