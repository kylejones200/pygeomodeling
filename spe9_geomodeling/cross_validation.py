"""
Cross-validation utilities with spatial awareness.

Provides spatial cross-validation methods specifically designed for geostatistical data.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from .exceptions import CrossValidationError, raise_invalid_parameter

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class SpatialKFold:
    """Spatial K-Fold cross-validation.

    Splits data based on spatial blocks to avoid spatial autocorrelation issues.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """Initialize spatial K-Fold.

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle blocks before splitting
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate spatial fold indices.

        Args:
            X: Feature array with spatial coordinates (first 3 columns: x, y, z)
            y: Target array (optional)
            groups: Group labels (optional)

        Returns:
            List of (train_idx, test_idx) tuples
        """
        X.shape[0]

        # Extract spatial coordinates (assuming first 3 columns are x, y, z)
        coords = X[:, :3]

        # Create spatial blocks based on coordinates
        # Divide space into n_splits^(1/3) blocks per dimension
        blocks_per_dim = int(np.ceil(self.n_splits ** (1 / 3)))

        # Compute block indices for each sample
        x_blocks = np.floor(coords[:, 0] * blocks_per_dim).astype(int)
        y_blocks = np.floor(coords[:, 1] * blocks_per_dim).astype(int)
        z_blocks = np.floor(coords[:, 2] * blocks_per_dim).astype(int)

        # Combine into single block ID
        block_ids = x_blocks * blocks_per_dim**2 + y_blocks * blocks_per_dim + z_blocks

        # Get unique blocks
        unique_blocks = np.unique(block_ids)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_blocks)

        # Split blocks into folds
        fold_sizes = np.full(self.n_splits, len(unique_blocks) // self.n_splits)
        fold_sizes[: len(unique_blocks) % self.n_splits] += 1

        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_blocks = unique_blocks[start:stop]
            test_mask = np.isin(block_ids, test_blocks)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            folds.append((train_idx, test_idx))
            current = stop

        return folds

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


class BlockCV:
    """Block cross-validation for spatial data.

    Divides the spatial domain into blocks and uses blocks as folds.
    """

    def __init__(
        self,
        n_blocks_x: int = 3,
        n_blocks_y: int = 3,
        n_blocks_z: int = 1,
        buffer_size: float = 0.0,
    ):
        """Initialize block cross-validation.

        Args:
            n_blocks_x: Number of blocks in x direction
            n_blocks_y: Number of blocks in y direction
            n_blocks_z: Number of blocks in z direction
            buffer_size: Buffer zone size (fraction of block size) to exclude
        """
        self.n_blocks_x = n_blocks_x
        self.n_blocks_y = n_blocks_y
        self.n_blocks_z = n_blocks_z
        self.buffer_size = buffer_size
        self.n_splits = n_blocks_x * n_blocks_y * n_blocks_z

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate block fold indices.

        Args:
            X: Feature array with spatial coordinates (first 3 columns: x, y, z)
            y: Target array (optional)
            groups: Group labels (optional)

        Returns:
            List of (train_idx, test_idx) tuples
        """
        coords = X[:, :3]

        # Normalize coordinates to [0, 1]
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-10)

        # Compute block indices
        x_blocks = np.floor(coords_norm[:, 0] * self.n_blocks_x).astype(int)
        y_blocks = np.floor(coords_norm[:, 1] * self.n_blocks_y).astype(int)
        z_blocks = np.floor(coords_norm[:, 2] * self.n_blocks_z).astype(int)

        # Clip to valid range
        x_blocks = np.clip(x_blocks, 0, self.n_blocks_x - 1)
        y_blocks = np.clip(y_blocks, 0, self.n_blocks_y - 1)
        z_blocks = np.clip(z_blocks, 0, self.n_blocks_z - 1)

        folds = []
        for ix in range(self.n_blocks_x):
            for iy in range(self.n_blocks_y):
                for iz in range(self.n_blocks_z):
                    # Test set: current block
                    test_mask = (x_blocks == ix) & (y_blocks == iy) & (z_blocks == iz)

                    # Apply buffer if specified
                    if self.buffer_size > 0:
                        # Exclude samples near block boundaries
                        block_width_x = 1.0 / self.n_blocks_x
                        block_width_y = 1.0 / self.n_blocks_y
                        block_width_z = 1.0 / self.n_blocks_z

                        buffer_x = block_width_x * self.buffer_size
                        buffer_y = block_width_y * self.buffer_size
                        buffer_z = block_width_z * self.buffer_size

                        x_min = ix * block_width_x + buffer_x
                        x_max = (ix + 1) * block_width_x - buffer_x
                        y_min = iy * block_width_y + buffer_y
                        y_max = (iy + 1) * block_width_y - buffer_y
                        z_min = iz * block_width_z + buffer_z
                        z_max = (iz + 1) * block_width_z - buffer_z

                        buffer_mask = (
                            (coords_norm[:, 0] >= x_min)
                            & (coords_norm[:, 0] <= x_max)
                            & (coords_norm[:, 1] >= y_min)
                            & (coords_norm[:, 1] <= y_max)
                            & (coords_norm[:, 2] >= z_min)
                            & (coords_norm[:, 2] <= z_max)
                        )

                        test_mask = test_mask & buffer_mask

                    test_idx = np.where(test_mask)[0]
                    train_idx = np.where(~test_mask)[0]

                    if len(test_idx) > 0 and len(train_idx) > 0:
                        folds.append((train_idx, test_idx))

        return folds

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Return the number of splits."""
        return len(self.split(X, y, groups)) if X is not None else self.n_splits


def cross_validate_spatial(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Any] = 5,
    scoring: Union[str, Callable] = "r2",
    return_train_score: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Perform spatial cross-validation.

    Args:
        model: Estimator to evaluate
        X: Feature array
        y: Target array
        cv: Cross-validation strategy (int or CV object)
        scoring: Scoring metric ('r2', 'mse', 'mae', or callable)
        return_train_score: Whether to return training scores
        verbose: Whether to show progress

    Returns:
        Dictionary with test scores (and train scores if requested)
    """
    # Create CV splitter
    if isinstance(cv, int):
        cv_splitter = SpatialKFold(n_splits=cv)
    else:
        cv_splitter = cv

    # Get scoring function
    if scoring == "r2":
        score_func = r2_score
    elif scoring == "mse":
        score_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    elif scoring == "mae":
        score_func = lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)
    elif callable(scoring):
        score_func = scoring
    else:
        raise_invalid_parameter("scoring", scoring, ["r2", "mse", "mae", "callable"])

    # Perform cross-validation
    test_scores = []
    train_scores = [] if return_train_score else None

    splits = cv_splitter.split(X, y)
    iterator = tqdm(splits, desc="CV Folds") if verbose else splits

    for train_idx, test_idx in iterator:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and train model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        # Evaluate
        y_pred = model_clone.predict(X_test)
        test_score = score_func(y_test, y_pred)
        test_scores.append(test_score)

        if return_train_score:
            y_train_pred = model_clone.predict(X_train)
            train_score = score_func(y_train, y_train_pred)
            train_scores.append(train_score)

    results = {"test_score": np.array(test_scores)}
    if return_train_score:
        results["train_score"] = np.array(train_scores)

    if verbose:
        print(f"\nCross-validation results:")
        print(f"  Test score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
        if return_train_score:
            print(
                f"  Train score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}"
            )

    return results


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""

    def __init__(
        self,
        model_class: type,
        param_space: dict[str, Any],
        cv: Union[int, Any] = 5,
        n_trials: int = 100,
        scoring: str = "r2",
        random_state: Optional[int] = None,
    ):
        """Initialize hyperparameter tuner.

        Args:
            model_class: Model class to tune
            param_space: Parameter search space
            cv: Cross-validation strategy
            n_trials: Number of optimization trials
            scoring: Scoring metric
            random_state: Random state for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise CrossValidationError(
                "Optuna is not installed",
                suggestion="Install Optuna: pip install optuna",
            )

        self.model_class = model_class
        self.param_space = param_space
        self.cv = cv
        self.n_trials = n_trials
        self.scoring = scoring
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.best_score = None

    def objective(self, trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna.

        Args:
            trial: Optuna trial
            X: Feature array
            y: Target array

        Returns:
            Score to maximize
        """
        # Sample parameters
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        # Create model with sampled parameters
        model = self.model_class(**params)

        # Perform cross-validation
        cv_results = cross_validate_spatial(
            model, X, y, cv=self.cv, scoring=self.scoring, verbose=False
        )

        return np.mean(cv_results["test_score"])

    def tune(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> dict[str, Any]:
        """Run hyperparameter tuning.

        Args:
            X: Feature array
            y: Target array
            verbose: Whether to show progress

        Returns:
            Dictionary with best parameters and score
        """
        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Optimize
        if verbose:
            print(f"Starting hyperparameter tuning with {self.n_trials} trials...")

        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=verbose,
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        if verbose:
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study": self.study,
        }

    def get_best_model(self) -> BaseEstimator:
        """Get model with best parameters.

        Returns:
            Model instance with best parameters
        """
        if self.best_params is None:
            raise CrossValidationError(
                "No tuning has been performed yet",
                suggestion="Call tune() first to find best parameters",
            )

        return self.model_class(**self.best_params)
