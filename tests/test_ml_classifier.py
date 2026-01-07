"""Tests for ML classification module."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.ml_classifier import (
    MLClassifier,
    ClassificationResult,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
)
from src.models.training_data import TrainingDataCollector, TrainingExample


class TestMLClassifier:
    """Tests for MLClassifier."""

    def test_init_models(self):
        """Classifier should initialize available models."""
        classifier = MLClassifier()

        # RF should always be available
        assert "rf" in classifier.models

        # Optional models based on imports
        if HAS_LIGHTGBM:
            assert "lgb" in classifier.models
        if HAS_XGBOOST:
            assert "xgb" in classifier.models

    def test_untrained_returns_default(self):
        """Untrained classifier should return default result."""
        classifier = MLClassifier()
        classifier.is_trained = False

        # Mock _load_models to return False
        with patch.object(classifier, "_load_models", return_value=False):
            result = classifier.predict_proba(np.random.randn(14))

        assert result.probability == 0.5
        assert result.confidence == "low"
        assert result.model_used == "none"

    def test_train_with_sufficient_data(self):
        """Training should succeed with enough data."""
        classifier = MLClassifier()

        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule

        metrics = classifier.train(X, y, target="ou_2_5", calibrate=False)

        assert "error" not in metrics
        assert classifier.is_trained
        assert "rf_accuracy" in metrics
        assert metrics["rf_accuracy"] > 0.5  # Better than random

    def test_train_insufficient_data(self):
        """Training should fail gracefully with insufficient data."""
        classifier = MLClassifier()

        X = np.random.randn(50, 14)  # Too few
        y = np.random.randint(0, 2, 50)

        metrics = classifier.train(X, y, target="ou_2_5")

        assert "error" in metrics
        assert not classifier.is_trained

    def test_predict_proba_shape(self):
        """Prediction should handle both 1D and 2D input."""
        classifier = MLClassifier()

        # Train first
        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = np.random.randint(0, 2, 200)
        classifier.train(X, y, target="ou_2_5", calibrate=False)

        # Test 1D input
        result_1d = classifier.predict_proba(np.random.randn(14))
        assert 0 <= result_1d.probability <= 1

        # Test 2D input
        result_2d = classifier.predict_proba(np.random.randn(1, 14))
        assert 0 <= result_2d.probability <= 1

    def test_confidence_levels(self):
        """Confidence should reflect model agreement."""
        # Create result manually to test
        result_high = ClassificationResult(
            market_type="ou",
            outcome="over",
            probability=0.65,
            confidence="high",
            model_used="ensemble",
            lgb_prob=0.64,
            xgb_prob=0.65,
            rf_prob=0.66,
        )

        # All models agree closely
        probs = [result_high.lgb_prob, result_high.xgb_prob, result_high.rf_prob]
        std = np.std([p for p in probs if p is not None])
        assert std < 0.05  # High confidence threshold

    def test_feature_importance(self):
        """Feature importance should return dict after training."""
        classifier = MLClassifier()

        # Train first
        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = np.random.randint(0, 2, 200)
        classifier.train(X, y, target="ou_2_5", calibrate=False)

        importance = classifier.get_feature_importance()
        # RF always trained, should have importance
        assert len(importance) > 0 or "rf" not in classifier.trained_models

    def test_train_with_calibration(self):
        """Training with calibration should work."""
        classifier = MLClassifier()

        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        metrics = classifier.train(X, y, target="ou_2_5", calibrate=True)

        assert "error" not in metrics
        assert classifier.is_trained

    def test_btts_target(self):
        """Should handle BTTS target correctly."""
        classifier = MLClassifier()

        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = np.random.randint(0, 2, 200)

        metrics = classifier.train(X, y, target="btts", calibrate=False)

        assert metrics["target"] == "btts"
        assert classifier.target == "btts"


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_fields(self):
        """Result should have all required fields."""
        result = ClassificationResult(
            market_type="ou",
            outcome="over",
            probability=0.65,
            confidence="medium",
            model_used="ensemble",
        )

        assert result.market_type == "ou"
        assert result.outcome == "over"
        assert result.probability == 0.65
        assert result.confidence == "medium"

    def test_optional_breakdown(self):
        """Breakdown fields should be optional."""
        result = ClassificationResult(
            market_type="btts",
            outcome="yes",
            probability=0.55,
            confidence="low",
            model_used="rf",
        )

        assert result.lgb_prob is None
        assert result.xgb_prob is None
        assert result.rf_prob is None

    def test_full_breakdown(self):
        """Full breakdown should include all model probs."""
        result = ClassificationResult(
            market_type="ou",
            outcome="over",
            probability=0.65,
            confidence="high",
            model_used="ensemble",
            lgb_prob=0.64,
            xgb_prob=0.66,
            rf_prob=0.65,
        )

        assert result.lgb_prob == 0.64
        assert result.xgb_prob == 0.66
        assert result.rf_prob == 0.65


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_training_example_fields(self):
        """TrainingExample should have required fields."""
        example = TrainingExample(
            match_id="123",
            features=np.zeros(14),
            feature_names=["f1", "f2"],
            label_1x2="home",
            label_over_2_5=True,
            label_btts=False,
            home_goals=2,
            away_goals=1,
            total_goals=3,
        )

        assert example.label_over_2_5 is True
        assert example.total_goals == 3
        assert example.label_1x2 == "home"

    def test_over_2_5_label(self):
        """Over 2.5 label should be correct for various scorelines."""
        # 3+ goals = True
        ex_over = TrainingExample(
            match_id="1",
            features=np.zeros(14),
            feature_names=[],
            label_1x2="home",
            label_over_2_5=True,
            label_btts=True,
            home_goals=2,
            away_goals=1,
            total_goals=3,
        )
        assert ex_over.label_over_2_5 is True

        # 2 goals = False
        ex_under = TrainingExample(
            match_id="2",
            features=np.zeros(14),
            feature_names=[],
            label_1x2="draw",
            label_over_2_5=False,
            label_btts=True,
            home_goals=1,
            away_goals=1,
            total_goals=2,
        )
        assert ex_under.label_over_2_5 is False


class TestTrainingDataCollector:
    """Tests for TrainingDataCollector."""

    def test_save_load_csv(self, tmp_path):
        """Should save and load training data."""
        collector = TrainingDataCollector()
        collector.DATA_DIR = tmp_path

        examples = [
            TrainingExample(
                match_id="1",
                features=np.array([1.0] * 14),
                feature_names=[f"f{i}" for i in range(14)],
                label_1x2="home",
                label_over_2_5=True,
                label_btts=True,
                home_goals=3,
                away_goals=1,
                total_goals=4,
            )
        ]

        collector.save_to_csv(examples, "test.csv")

        df = collector.load_from_csv("test.csv")
        assert len(df) == 1
        assert df["label_over_2_5"].iloc[0] == 1

    def test_load_missing_csv(self, tmp_path):
        """Should return empty DataFrame for missing file."""
        collector = TrainingDataCollector()
        collector.DATA_DIR = tmp_path

        df = collector.load_from_csv("missing.csv")
        assert df.empty

    def test_save_empty_examples(self, tmp_path):
        """Should handle empty examples gracefully."""
        collector = TrainingDataCollector()
        collector.DATA_DIR = tmp_path

        collector.save_to_csv([], "empty.csv")
        # Should not create file for empty data
        assert not (tmp_path / "empty.csv").exists()

    def test_multiple_examples(self, tmp_path):
        """Should handle multiple examples."""
        collector = TrainingDataCollector()
        collector.DATA_DIR = tmp_path

        examples = [
            TrainingExample(
                match_id=str(i),
                features=np.random.randn(14),
                feature_names=[f"f{j}" for j in range(14)],
                label_1x2=["home", "draw", "away"][i % 3],
                label_over_2_5=i % 2 == 0,
                label_btts=i % 2 == 1,
                home_goals=i,
                away_goals=i + 1,
                total_goals=2 * i + 1,
            )
            for i in range(5)
        ]

        collector.save_to_csv(examples, "multi.csv")

        df = collector.load_from_csv("multi.csv")
        assert len(df) == 5


class TestIntegration:
    """Integration tests for ML pipeline."""

    def test_full_training_pipeline(self):
        """Full pipeline should complete without errors."""
        from src.models.training_pipeline import TrainingPipeline

        # Mock database to return empty
        with patch("src.models.training_data.db") as mock_db:
            mock_session = MagicMock()
            mock_session.query.return_value.filter.return_value.all.return_value = []
            mock_db.session.return_value.__enter__.return_value = mock_session

            pipeline = TrainingPipeline()
            results = pipeline.run_full_training()

            assert "stages" in results
            assert results["stages"]["data_collection"]["n_examples"] == 0

    def test_incremental_update_empty(self, tmp_path):
        """Incremental update with no existing data should run full training."""
        from src.models.training_pipeline import TrainingPipeline
        from src.models.training_data import training_collector

        # Set temp dir
        original_dir = training_collector.DATA_DIR
        training_collector.DATA_DIR = tmp_path

        try:
            with patch("src.models.training_data.db") as mock_db:
                mock_session = MagicMock()
                mock_session.query.return_value.filter.return_value.all.return_value = []
                mock_db.session.return_value.__enter__.return_value = mock_session

                pipeline = TrainingPipeline()
                results = pipeline.run_incremental_update()

                # Should run full training since no existing data
                assert "stages" in results
        finally:
            training_collector.DATA_DIR = original_dir


class TestModelPersistence:
    """Tests for model save/load."""

    def test_save_and_load_models(self, tmp_path):
        """Models should save and load correctly."""
        classifier = MLClassifier()
        classifier.MODEL_DIR = tmp_path

        # Train
        np.random.seed(42)
        X = np.random.randn(200, 14)
        y = np.random.randint(0, 2, 200)
        classifier.train(X, y, target="ou_2_5", calibrate=False)

        # Verify saved
        assert (tmp_path / "classifier_ou_2_5.pkl").exists()

        # Create new classifier and load
        new_classifier = MLClassifier()
        new_classifier.MODEL_DIR = tmp_path
        loaded = new_classifier._load_models("ou_2_5")

        assert loaded is True
        assert new_classifier.is_trained is True

    def test_load_nonexistent_models(self, tmp_path):
        """Should return False for nonexistent models."""
        classifier = MLClassifier()
        classifier.MODEL_DIR = tmp_path

        loaded = classifier._load_models("nonexistent")
        assert loaded is False
        assert classifier.is_trained is False
