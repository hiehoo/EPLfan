"""Training pipeline for ML models."""
import logging
from datetime import datetime
from pathlib import Path
import json

import numpy as np

from src.models.training_data import training_collector
from src.models.ml_classifier import ml_classifier_ou, ml_classifier_btts

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline."""

    RESULTS_DIR = Path("data/training/results")

    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_full_training(self) -> dict:
        """Run full training pipeline."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stages": {},
        }

        # Stage 1: Collect data
        logger.info("Stage 1: Collecting training data...")
        examples = training_collector.collect_from_database()
        results["stages"]["data_collection"] = {
            "n_examples": len(examples),
            "status": "success" if len(examples) >= 100 else "insufficient_data",
        }

        if len(examples) < 100:
            logger.warning("Insufficient training data")
            return results

        # Save to CSV for debugging
        training_collector.save_to_csv(examples)

        # Stage 2: Prepare features
        logger.info("Stage 2: Preparing features...")
        X = np.array([ex.features for ex in examples])
        y_ou = np.array([int(ex.label_over_2_5) for ex in examples])
        y_btts = np.array([int(ex.label_btts) for ex in examples])

        results["stages"]["feature_preparation"] = {
            "n_features": X.shape[1],
            "ou_base_rate": float(np.mean(y_ou)),
            "btts_base_rate": float(np.mean(y_btts)),
        }

        # Stage 3: Train O/U model
        logger.info("Stage 3: Training O/U 2.5 classifier...")
        ou_metrics = ml_classifier_ou.train(X, y_ou, target="ou_2_5")
        results["stages"]["ou_training"] = ou_metrics

        # Stage 4: Train BTTS model
        logger.info("Stage 4: Training BTTS classifier...")
        btts_metrics = ml_classifier_btts.train(X, y_btts, target="btts")
        results["stages"]["btts_training"] = btts_metrics

        # Stage 5: Feature importance
        logger.info("Stage 5: Analyzing feature importance...")
        ou_importance = ml_classifier_ou.get_feature_importance()
        results["stages"]["feature_importance"] = {
            "ou_top_5": sorted(ou_importance.items(), key=lambda x: -x[1])[:5] if ou_importance else [],
        }

        # Save results
        self._save_results(results)

        return results

    def run_incremental_update(self) -> dict:
        """Update models with new data."""
        # Load existing data
        df = training_collector.load_from_csv()

        if df.empty:
            return self.run_full_training()

        # Get new examples
        new_examples = training_collector.collect_from_database()

        # Filter to only new ones
        existing_ids = set(df["match_id"].astype(str))
        new_examples = [ex for ex in new_examples if ex.match_id not in existing_ids]

        if len(new_examples) < 10:
            logger.info("No significant new data, skipping update")
            return {"status": "skipped", "reason": "no_new_data"}

        # Append and retrain
        training_collector.save_to_csv(
            training_collector.collect_from_database()
        )

        return self.run_full_training()

    def _save_results(self, results: dict):
        """Save training results."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = self.RESULTS_DIR / f"training_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")


# Singleton
training_pipeline = TrainingPipeline()
