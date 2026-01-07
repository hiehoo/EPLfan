"""Training pipeline for ML models."""
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.models.training_data import training_collector
from src.models.ml_classifier import ml_classifier_ou, ml_classifier_btts, MLClassifier

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

    def retrain_with_validation(
        self,
        target: str = "all",
        min_samples: int = 100,
        force_save: bool = False,
    ) -> dict:
        """Retrain models with holdout validation.

        Args:
            target: Target market ("ou", "btts", "all")
            min_samples: Minimum training samples required
            force_save: Save new model even if worse

        Returns:
            Dict with comparison metrics and status
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "status": "pending",
        }

        # Collect training data
        examples = training_collector.collect_from_database()
        if len(examples) < min_samples:
            results["status"] = "error"
            results["reason"] = "insufficient_data"
            results["n_examples"] = len(examples)
            return results

        # Prepare features
        X = np.array([ex.features for ex in examples])
        y_ou = np.array([int(ex.label_over_2_5) for ex in examples])
        y_btts = np.array([int(ex.label_btts) for ex in examples])

        # Train/test split
        X_train, X_test, y_ou_train, y_ou_test = train_test_split(
            X, y_ou, test_size=0.2, random_state=42, stratify=y_ou
        )
        _, _, y_btts_train, y_btts_test = train_test_split(
            X, y_btts, test_size=0.2, random_state=42, stratify=y_btts
        )

        results["n_train"] = len(X_train)
        results["n_test"] = len(X_test)

        # Process targets
        targets = ["ou", "btts"] if target == "all" else [target]
        target_results = {}

        for t in targets:
            y_train = y_ou_train if t == "ou" else y_btts_train
            y_test = y_ou_test if t == "ou" else y_btts_test
            classifier = ml_classifier_ou if t == "ou" else ml_classifier_btts
            target_name = "ou_2_5" if t == "ou" else "btts"

            # Evaluate old model (if exists)
            old_accuracy = None
            if classifier._load_models(target_name):
                old_preds = []
                for x in X_test:
                    pred = classifier.predict_proba(x)
                    old_preds.append(1 if pred.probability > 0.5 else 0)
                old_accuracy = accuracy_score(y_test, old_preds)

            # Train new model
            new_classifier = MLClassifier()
            new_classifier.train(X_train, y_train, target=target_name)

            # Evaluate new model
            new_preds = []
            for x in X_test:
                pred = new_classifier.predict_proba(x)
                new_preds.append(1 if pred.probability > 0.5 else 0)
            new_accuracy = accuracy_score(y_test, new_preds)

            # Compare and decide
            improvement = new_accuracy - (old_accuracy or 0.5)
            should_save = new_accuracy >= (old_accuracy or 0) or force_save

            if should_save:
                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d")
                versioned_path = new_classifier.MODEL_DIR / f"classifier_{target_name}_{timestamp}.pkl"
                main_path = new_classifier.MODEL_DIR / f"classifier_{target_name}.pkl"

                with open(versioned_path, "wb") as f:
                    pickle.dump({
                        "models": new_classifier.trained_models,
                        "target": target_name,
                    }, f)

                with open(main_path, "wb") as f:
                    pickle.dump({
                        "models": new_classifier.trained_models,
                        "target": target_name,
                    }, f)

                status = "saved"
                logger.info(f"Saved new {t} model (accuracy: {new_accuracy:.3f})")
            else:
                status = "kept_old"
                logger.info(f"Kept old {t} model (new: {new_accuracy:.3f} < old: {old_accuracy:.3f})")

            target_results[t] = {
                "status": status,
                "old_accuracy": round(old_accuracy, 4) if old_accuracy else None,
                "new_accuracy": round(new_accuracy, 4),
                "improvement": round(improvement, 4),
            }

        results["targets"] = target_results
        results["status"] = "completed"

        # Save results
        self._save_results(results)

        return results


# Singleton
training_pipeline = TrainingPipeline()
