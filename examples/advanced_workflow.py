"""
Advanced Workflow Example

Demonstrates the new advanced features:
- Model serialization with versioning
- Spatial cross-validation
- Hyperparameter tuning with Optuna
- Parallel model training
- Comprehensive error handling
"""

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


def main():
    print("=" * 80)
    print("PyGeomodeling Advanced Workflow Example")
    print("=" * 80)

    # =========================================================================
    # 1. Load Data with Error Handling
    # =========================================================================
    print("\n[1/7] Loading SPE9 dataset...")
    try:
        data = load_spe9_data()
        print(f"✓ Loaded data with dimensions: {data['dimensions']}")
        print(f"✓ Available properties: {list(data['properties'].keys())}")
    except exceptions.DataLoadError as e:
        print(f"Error loading data: {e}")
        return
    except exceptions.FileFormatError as e:
        print(f"Invalid file format: {e}")
        return

    # =========================================================================
    # 2. Prepare Features
    # =========================================================================
    print("\n[2/7] Preparing features...")
    toolkit = UnifiedSPE9Toolkit()
    toolkit.load_spe9_data(data)
    X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
        test_size=0.2, random_state=42
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {X_train.shape[1]}")

    # =========================================================================
    # 3. Spatial Cross-Validation
    # =========================================================================
    print("\n[3/7] Performing spatial cross-validation...")

    # Test with simple model first
    simple_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    # Spatial K-Fold
    print("\n  Spatial K-Fold (5 folds):")
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
    print(
        f"    Test R²: {results_spatial['test_score'].mean():.4f} ± {results_spatial['test_score'].std():.4f}"
    )
    print(
        f"    Train R²: {results_spatial['train_score'].mean():.4f} ± {results_spatial['train_score'].std():.4f}"
    )

    # Block CV
    print("\n  Block Cross-Validation (3x3x1 blocks):")
    cv_block = BlockCV(n_blocks_x=3, n_blocks_y=3, n_blocks_z=1, buffer_size=0.05)
    results_block = cross_validate_spatial(
        model=simple_model, X=X_train, y=y_train, cv=cv_block, verbose=False
    )
    print(
        f"    Test R²: {results_block['test_score'].mean():.4f} ± {results_block['test_score'].std():.4f}"
    )

    # =========================================================================
    # 4. Hyperparameter Tuning (Optional - requires Optuna)
    # =========================================================================
    print("\n[4/7] Hyperparameter tuning...")
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
        print("  Running Optuna optimization (20 trials)...")
        tuning_results = tuner.tune(X_train, y_train, verbose=False)

        print(f"  ✓ Best parameters: {tuning_results['best_params']}")
        print(f"  ✓ Best CV score: {tuning_results['best_score']:.4f}")

        # Get best model
        best_rf = tuner.get_best_model()

    except exceptions.CrossValidationError as e:
        print(f"  Optuna not available: {e.suggestion}")
        print("  Using default Random Forest parameters...")
        best_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # =========================================================================
    # 5. Parallel Model Training
    # =========================================================================
    print("\n[5/7] Training multiple models in parallel...")

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

    print(f"\n  ✓ Trained {len(models)} models in {training_time:.2f}s")
    print("\n  Model Performance:")
    for name, result in sorted(
        results.items(), key=lambda x: x[1]["metrics"]["r2"], reverse=True
    ):
        print(f"    {name}:")
        print(f"      R²:  {result['metrics']['r2']:.4f}")
        print(f"      MSE: {result['metrics']['mse']:.4f}")
        print(f"      MAE: {result['metrics']['mae']:.4f}")
        print(f"      Time: {result['training_time']:.2f}s")

    # =========================================================================
    # 6. Model Serialization
    # =========================================================================
    print("\n[6/7] Saving models with metadata...")

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

    print(f"  ✓ Saved best model: {best_name}")
    print(f"    Location: {model_path}")
    print(f"    R² score: {best_metrics['r2']:.4f}")

    # Demonstrate loading
    print("\n  Loading saved model...")
    loaded_model, metadata, scaler = load_model(
        f"production_{best_name}", save_dir=save_dir
    )
    print(f"  ✓ Loaded model: {metadata.model_name}")
    print(f"    Version: {metadata.version}")
    print(f"    Created: {metadata.created_at}")

    # =========================================================================
    # 7. Batch Predictions
    # =========================================================================
    print("\n[7/7] Making batch predictions...")

    # Create batch predictor
    predictor = BatchPredictor(n_jobs=-1, batch_size=500, verbose=False)

    # Make predictions
    start_time = time.time()
    predictions = predictor.predict(loaded_model, X_test)
    pred_time = time.time() - start_time

    print(f"  ✓ Made {len(predictions)} predictions in {pred_time:.3f}s")
    print(f"    Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    # Predict with multiple models
    print("\n  Predicting with all models...")
    all_predictions = predictor.predict_multiple_models(
        {name: res["model"] for name, res in results.items()}, X_test[:1000]
    )

    print(f"  ✓ Generated predictions from {len(all_predictions)} models")

    # Compare predictions
    print("\n  Prediction statistics (first 1000 samples):")
    for name, preds in all_predictions.items():
        print(f"    {name}: mean={preds.mean():.2f}, std={preds.std():.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Workflow Complete!")
    print("=" * 80)
    print(f"\nBest Model: {best_name}")
    print(f"  R² Score: {best_metrics['r2']:.4f}")
    print(f"  MSE: {best_metrics['mse']:.4f}")
    print(f"  MAE: {best_metrics['mae']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print("\nKey Features Demonstrated:")
    print("  ✓ Spatial cross-validation")
    print("  ✓ Hyperparameter tuning (Optuna)")
    print("  ✓ Parallel model training")
    print("  ✓ Model serialization with metadata")
    print("  ✓ Batch predictions")
    print("  ✓ Comprehensive error handling")


if __name__ == "__main__":
    main()
