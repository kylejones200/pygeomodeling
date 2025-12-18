"""
Advanced Workflow Example.

Demonstrates the new advanced features:
- Model serialization with versioning
- Spatial cross-validation
- Hyperparameter tuning with Optuna
- Parallel model training
- Comprehensive error handling
"""

import logging
import time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from spe9_geomodeling import (
    BatchPredictor,
    BlockCV,
    HyperparameterTuner,
    ParallelModelTrainer,
    SpatialKFold,
    UnifiedSPE9Toolkit,
    cross_validate_spatial,
    exceptions,
    load_model,
    load_spe9_data,
    save_model,
)

# Configure logging
logger = logging.getLogger(__name__)


def main():
    """Run the advanced workflow example."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("PyGeomodeling Advanced Workflow Example")

    # =========================================================================
    # 1. Load Data with Error Handling
    # =========================================================================
    logger.info("[1/7] Loading SPE9 dataset...")
    try:
        data = load_spe9_data()
        logger.info("Loaded data with dimensions: %s", data['dimensions'])
        logger.info("Available properties: %s", list(data['properties'].keys()))
    except exceptions.DataLoadError as e:
        logger.error("Error loading data: %s", e)
        return
    except exceptions.FileFormatError as e:
        logger.error("Invalid file format: %s", e)
        return

    # =========================================================================
    # 2. Prepare Features
    # =========================================================================
    logger.info("[2/7] Preparing features...")
    toolkit = UnifiedSPE9Toolkit()
    toolkit.load_spe9_data(data)
    X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
        test_size=0.2, random_state=42
    )
    logger.info("Training samples: %d", len(X_train))
    logger.info("Test samples: %d", len(X_test))
    logger.info("Features: %d", X_train.shape[1])

    # =========================================================================
    # 3. Spatial Cross-Validation
    # =========================================================================
    logger.info("[3/7] Performing spatial cross-validation...")

    # Test with simple model first
    simple_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    # Spatial K-Fold
    logger.info("  Spatial K-Fold (5 folds):")
    cv_spatial = SpatialKFold(n_splits=5, shuffle=True, random_state=42)
    results_spatial = cross_validate_spatial(
        model=simple_model,
        X=X_train,
        y=y_train,
        cv=cv_spatial,
        scoring="r2",
        return_train_score=True,
        verbose=False,
    )
    logger.info("    Test R²: %.4f ± %.4f", results_spatial['test_score'].mean(), results_spatial['test_score'].std())
    logger.info("    Train R²: %.4f ± %.4f", results_spatial['train_score'].mean(), results_spatial['train_score'].std())

    # Block CV
    logger.info("  Block Cross-Validation (3x3x1 blocks):")
    cv_block = BlockCV(n_blocks_x=3, n_blocks_y=3, n_blocks_z=1, buffer_size=0.05)
    results_block = cross_validate_spatial(
        model=simple_model, X=X_train, y=y_train, cv=cv_block, verbose=False
    )
    logger.info("    Test R²: %.4f ± %.4f", results_block['test_score'].mean(), results_block['test_score'].std())

    # =========================================================================
    # 4. Hyperparameter Tuning (Optional - requires Optuna)
    # =========================================================================
    logger.info("[4/7] Hyperparameter tuning...")
    try:
        # Define search space
        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 5, "high": 15},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
        }

        # Create tuner
        tuner = HyperparameterTuner(
            model_class=RandomForestRegressor,
            param_space=param_space,
            cv=3,  # Use fewer folds for speed
            n_trials=20,  # Use fewer trials for demo
            scoring="r2",
            random_state=42,
        )

        # Run tuning
        logger.info("  Running Optuna optimization (20 trials)...")
        tuning_results = tuner.tune(X_train, y_train, verbose=False)

        logger.info("  Best parameters: %s", tuning_results['best_params'])
        logger.info("  Best CV score: %.4f", tuning_results['best_score'])

        # Get best model
        best_rf = tuner.get_best_model()

    except exceptions.CrossValidationError as e:
        logger.warning("  Optuna not available: %s", e.suggestion)
        logger.info("  Using default Random Forest parameters...")
        best_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # =========================================================================
    # 5. Parallel Model Training
    # =========================================================================
    logger.info("[5/7] Training multiple models in parallel...")

    # Define models to compare
    models = {
        "random_forest_tuned": best_rf,
        "random_forest_default": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "gpr_rbf": GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(), random_state=42
        ),
        "gpr_matern": GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern() + WhiteKernel(), random_state=42
        ),
    }

    # Train all models in parallel
    trainer = ParallelModelTrainer(n_jobs=-1, verbose=0)
    start_time = time.time()
    results = trainer.train_and_evaluate(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    training_time = time.time() - start_time

    logger.info("  Trained %d models in %.2fs", len(models), training_time)
    logger.info("  Model Performance:")
    for name, result in sorted(
        results.items(), key=lambda x: x[1]["metrics"]["r2"], reverse=True
    ):
        logger.info("    %s:", name)
        logger.info("      R²:  %.4f", result['metrics']['r2'])
        logger.info("      MSE: %.4f", result['metrics']['mse'])
        logger.info("      MAE: %.4f", result['metrics']['mae'])
        logger.info("      Time: %.2fs", result['training_time'])

    # =========================================================================
    # 6. Model Serialization
    # =========================================================================
    logger.info("[6/7] Saving models with metadata...")

    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["r2"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    # Save best model
    save_dir = Path("saved_models")
    model_path = save_model(
        model=best_model,
        model_name=f"production_{best_name}",
        model_type=best_name,
        backend="sklearn",
        save_dir=save_dir,
        metrics=best_metrics,
        description="Best model from advanced workflow example",
        dataset="SPE9",
        training_samples=len(X_train),
    )

    logger.info("  Saved best model: %s", best_name)
    logger.info("    Location: %s", model_path)
    logger.info("    R² score: %.4f", best_metrics['r2'])

    # Demonstrate loading
    logger.info("  Loading saved model...")
    loaded_model, metadata, scaler = load_model(
        f"production_{best_name}", save_dir=save_dir
    )
    logger.info("  Loaded model: %s", metadata.model_name)
    logger.info("    Version: %s", metadata.version)
    logger.info("    Created: %s", metadata.created_at)

    # =========================================================================
    # 7. Batch Predictions
    # =========================================================================
    logger.info("[7/7] Making batch predictions...")

    # Create batch predictor
    predictor = BatchPredictor(n_jobs=-1, batch_size=500, verbose=False)

    # Make predictions
    start_time = time.time()
    predictions = predictor.predict(loaded_model, X_test)
    pred_time = time.time() - start_time

    logger.info("  Made %d predictions in %.3fs", len(predictions), pred_time)
    logger.info("    Prediction range: [%.2f, %.2f]", predictions.min(), predictions.max())

    # Predict with multiple models
    logger.info("  Predicting with all models...")
    all_predictions = predictor.predict_multiple_models(
        {name: res["model"] for name, res in results.items()}, X_test[:1000]
    )

    logger.info("  Generated predictions from %d models", len(all_predictions))

    # Compare predictions
    logger.info("  Prediction statistics (first 1000 samples):")
    for name, preds in all_predictions.items():
        logger.info("    %s: mean=%.2f, std=%.2f", name, preds.mean(), preds.std())

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("Workflow Complete!")
    logger.info("Best Model: %s", best_name)
    logger.info("  R² Score: %.4f", best_metrics['r2'])
    logger.info("  MSE: %.4f", best_metrics['mse'])
    logger.info("  MAE: %.4f", best_metrics['mae'])
    logger.info("Model saved to: %s", model_path)
    logger.info("Key Features Demonstrated:")
    logger.info("  - Spatial cross-validation")
    logger.info("  - Hyperparameter tuning (Optuna)")
    logger.info("  - Parallel model training")
    logger.info("  - Model serialization with metadata")
    logger.info("  - Batch predictions")
    logger.info("  - Comprehensive error handling")


if __name__ == "__main__":
    main()
