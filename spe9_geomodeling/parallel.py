"""
Parallel processing utilities for model training and prediction.

Leverages joblib for efficient parallel computation.
"""

from typing import Any, Callable, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm


class ParallelModelTrainer:
    """Train multiple models in parallel."""

    def __init__(self, n_jobs: int = -1, verbose: int = 1):
        """Initialize parallel trainer.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    def train_models(
        self,
        models: dict[str, BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, BaseEstimator]:
        """Train multiple models in parallel.

        Args:
            models: Dictionary of model_name -> model instance
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary of model_name -> trained model
        """
        if self.verbose > 0:
            print(
                f"Training {len(models)} models in parallel (n_jobs={self.n_jobs})..."
            )

        def train_single_model(
            name: str, model: BaseEstimator
        ) -> tuple[str, BaseEstimator]:
            """Train a single model."""
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            return name, model_clone

        # Train models in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(train_single_model)(name, model) for name, model in models.items()
        )

        # Convert to dictionary
        trained_models = dict(results)

        if self.verbose > 0:
            print(f"✓ Trained {len(trained_models)} models")

        return trained_models

    def train_and_evaluate(
        self,
        models: dict[str, BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metrics: dict[str, Callable] = None,
    ) -> dict[str, dict[str, Any]]:
        """Train and evaluate multiple models in parallel.

        Args:
            models: Dictionary of model_name -> model instance
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            metrics: Dictionary of metric_name -> metric function

        Returns:
            Dictionary of model_name -> results dict
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if metrics is None:
            metrics = {
                "r2": r2_score,
                "mse": mean_squared_error,
                "mae": mean_absolute_error,
            }

        if self.verbose > 0:
            print(
                f"Training and evaluating {len(models)} models in parallel (n_jobs={self.n_jobs})..."
            )

        def train_and_eval_single(
            name: str, model: BaseEstimator
        ) -> tuple[str, dict[str, Any]]:
            """Train and evaluate a single model."""
            import time

            start_time = time.time()

            # Train
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predict
            y_pred = model_clone.predict(X_test)

            # Evaluate
            results = {
                "model": model_clone,
                "predictions": y_pred,
                "training_time": time.time() - start_time,
                "metrics": {},
            }

            for metric_name, metric_func in metrics.items():
                results["metrics"][metric_name] = metric_func(y_test, y_pred)

            return name, results

        # Train and evaluate in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(train_and_eval_single)(name, model)
            for name, model in models.items()
        )

        # Convert to dictionary
        all_results = dict(results)

        if self.verbose > 0:
            print(f"\n✓ Results for {len(all_results)} models:")
            for name, result in all_results.items():
                print(f"  {name}:")
                for metric_name, metric_value in result["metrics"].items():
                    print(f"    {metric_name}: {metric_value:.4f}")
                print(f"    training_time: {result['training_time']:.2f}s")

        return all_results


class BatchPredictor:
    """Make predictions in parallel batches."""

    def __init__(self, n_jobs: int = -1, batch_size: int = 1000, verbose: bool = True):
        """Initialize batch predictor.

        Args:
            n_jobs: Number of parallel jobs
            batch_size: Size of each batch
            verbose: Whether to show progress
        """
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose

    def predict(
        self, model: BaseEstimator, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Make predictions in parallel batches.

        Args:
            model: Trained model
            X: Features to predict
            return_std: Whether to return standard deviations (for GP models)

        Returns:
            Predictions (and standard deviations if requested)
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))

        if self.verbose:
            print(
                f"Making predictions for {n_samples} samples in {n_batches} batches..."
            )

        def predict_batch(
            batch_idx: int,
        ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
            """Predict a single batch."""
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]

            if (
                return_std
                and hasattr(model, "predict")
                and "return_std" in str(model.predict.__code__.co_varnames)
            ):
                return model.predict(X_batch, return_std=True)
            else:
                return model.predict(X_batch)

        # Predict batches in parallel
        if self.verbose:
            results = []
            for batch_idx in tqdm(range(n_batches), desc="Predicting"):
                results.append(predict_batch(batch_idx))
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_batch)(i) for i in range(n_batches)
            )

        # Combine results
        if return_std and isinstance(results[0], tuple):
            predictions = np.concatenate([r[0] for r in results])
            std_devs = np.concatenate([r[1] for r in results])
            return predictions, std_devs
        else:
            predictions = np.concatenate(results)
            return predictions

    def predict_multiple_models(
        self, models: dict[str, BaseEstimator], X: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Make predictions with multiple models in parallel.

        Args:
            models: Dictionary of model_name -> model
            X: Features to predict

        Returns:
            Dictionary of model_name -> predictions
        """
        if self.verbose:
            print(f"Making predictions with {len(models)} models...")

        def predict_single_model(
            name: str, model: BaseEstimator
        ) -> tuple[str, np.ndarray]:
            """Predict with a single model."""
            predictions = self.predict(model, X, return_std=False)
            return name, predictions

        # Predict with all models in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_single_model)(name, model) for name, model in models.items()
        )

        return dict(results)


class ParallelCrossValidator:
    """Perform cross-validation with parallel fold evaluation."""

    def __init__(self, n_jobs: int = -1, verbose: bool = True):
        """Initialize parallel cross-validator.

        Args:
            n_jobs: Number of parallel jobs
            verbose: Whether to show progress
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: Any,
        scoring: Callable = None,
    ) -> dict[str, np.ndarray]:
        """Perform cross-validation with parallel fold evaluation.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_splitter: Cross-validation splitter
            scoring: Scoring function

        Returns:
            Dictionary with scores
        """
        from sklearn.metrics import r2_score

        if scoring is None:
            scoring = r2_score

        splits = list(cv_splitter.split(X, y))

        if self.verbose:
            print(f"Performing {len(splits)}-fold cross-validation...")

        def evaluate_fold(
            fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray
        ) -> float:
            """Evaluate a single fold."""
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predict and score
            y_pred = model_clone.predict(X_test)
            score = scoring(y_test, y_pred)

            return score

        # Evaluate folds in parallel
        if self.verbose:
            scores = []
            for i, (train_idx, test_idx) in enumerate(tqdm(splits, desc="CV Folds")):
                score = evaluate_fold(i, train_idx, test_idx)
                scores.append(score)
        else:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(evaluate_fold)(i, train_idx, test_idx)
                for i, (train_idx, test_idx) in enumerate(splits)
            )

        scores = np.array(scores)

        if self.verbose:
            print(f"\nCross-validation score: {scores.mean():.4f} ± {scores.std():.4f}")

        return {
            "test_scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
        }


def parallel_grid_search(
    model_class: type,
    param_grid: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scoring: Callable = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Perform parallel grid search over hyperparameters.

    Args:
        model_class: Model class to instantiate
        param_grid: Dictionary of parameter_name -> list of values
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        scoring: Scoring function
        n_jobs: Number of parallel jobs
        verbose: Whether to show progress

    Returns:
        Dictionary with best parameters and results
    """
    from itertools import product

    from sklearn.metrics import r2_score

    if scoring is None:
        scoring = r2_score

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations...")

    def evaluate_params(params_tuple: tuple) -> tuple[dict[str, Any], float]:
        """Evaluate a single parameter combination."""
        params = dict(zip(param_names, params_tuple))

        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        score = scoring(y_test, y_pred)

        return params, score

    # Evaluate all combinations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_params)(params) for params in param_combinations
    )

    # Find best parameters
    best_params, best_score = max(results, key=lambda x: x[1])

    if verbose:
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results,
    }
