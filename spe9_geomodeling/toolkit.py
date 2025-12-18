"""SPE9 Geomodeling Toolkit.

A unified, Pythonic toolkit for reservoir property modeling using the SPE9 dataset.
Supports both scikit-learn and GPyTorch workflows in a single interface.

.. deprecated:: 0.3.0
    This module is deprecated. Use :class:`UnifiedSPE9Toolkit` from
    :mod:`spe9_geomodeling.unified_toolkit` instead.
"""

from __future__ import annotations

import logging
import warnings

warnings.warn(
    "spe9_geomodeling.toolkit.SPE9Toolkit is deprecated. "
    "Use spe9_geomodeling.UnifiedSPE9Toolkit instead.",
    DeprecationWarning,
    stacklevel=2,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import GPyTorch components (optional)
try:
    import gpytorch
    import torch

    from .model_gp import GPModel

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    torch = None
    gpytorch = None
    GPModel = None

from .grdecl_parser import load_spe9_data

warnings.filterwarnings("ignore")


@dataclass
class ModelResults:
    """Container for model evaluation results."""

    r2: float
    rmse: float
    mae: float
    y_pred: np.ndarray
    y_std: np.ndarray | None = None


@dataclass
class GridData:
    """Container for grid data and features."""

    X_grid: np.ndarray
    y_grid: np.ndarray
    feature_names: list[str]
    permx_3d: np.ndarray
    dimensions: tuple[int, int, int]
    X_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    X_train_scaled: np.ndarray | None = None
    y_train_scaled: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


class SPE9Toolkit:
    """Unified SPE9 Geomodeling Toolkit.

    A comprehensive toolkit that provides explicit control over each step of the
    geomodeling workflow. Supports both scikit-learn and GPyTorch backends.

    Attributes:
        data_path: Path to SPE9 dataset file
        backend: Modeling backend ('sklearn' or 'gpytorch')
        data: Loaded SPE9 dataset
        grid_data: Container for grid data and features
        models: Dictionary of trained models
        scalers: Dictionary of data scalers
        results: Dictionary of model evaluation results
    """

    def __init__(
        self, data_path: str | Path | None = None, backend: str = "sklearn"
    ) -> None:
        """Initialize the toolkit.

        Args:
            data_path: Path to SPE9 dataset file. If None, uses bundled data.
            backend: Modeling backend ('sklearn' or 'gpytorch')

        Raises:
            ValueError: If backend is invalid or GPyTorch backend is requested but not available
        """
        # Validate backend
        if backend not in ["sklearn", "gpytorch"]:
            raise ValueError("Backend must be 'sklearn' or 'gpytorch'")

        if backend == "gpytorch" and not GPYTORCH_AVAILABLE:
            raise ValueError(
                "GPyTorch backend requested but GPyTorch is not installed. "
                "Install with: pip install torch gpytorch"
            )

        # Set up data path
        if data_path is None:
            # Use the bundled data file in the project
            module_dir = Path(__file__).parent.parent
            default_path = module_dir / "data" / "SPE9.GRDECL"
        else:
            default_path = Path(data_path)

        self.data_path = default_path
        self.backend = backend

        # Initialize data containers
        self.data: dict[str, Any] | None = None
        self.grid_data: GridData | None = None

        # Model management
        self.models: dict[str, BaseEstimator] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.results: dict[str, ModelResults] = {}

        logger.info("SPE9 Toolkit initialized with %s backend", backend)

    def load_data(self) -> dict[str, Any]:
        """Load SPE9 dataset.

        Returns:
            Dictionary containing the loaded SPE9 data

        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"SPE9 data file not found: {self.data_path}")

        logger.info("Loading SPE9 dataset from %s", self.data_path)
        self.data = load_spe9_data(str(self.data_path))

        nx, ny, nz = self.data["dimensions"]
        permx_3d = self.data["properties"]["PERMX"]

        logger.info("Grid dimensions: %d x %d x %d", nx, ny, nz)
        logger.info("PERMX range: %.2f - %.2f mD", permx_3d.min(), permx_3d.max())
        logger.info("PERMX mean: %.2f mD", permx_3d.mean())

        return self.data

    def prepare_features(
        self, *, add_geological_features: bool = False, log_transform: bool = True
    ) -> GridData:
        """Prepare features for modeling.

        Args:
            add_geological_features: Whether to add geological features
            log_transform: Whether to apply log transform to permeability

        Returns:
            GridData container with features and targets

        Raises:
            ValueError: If data hasn't been loaded
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        nx, ny, nz = self.data["dimensions"]
        permx_3d = self.data["properties"]["PERMX"]

        # Create coordinate grids
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )

        # Flatten coordinates
        X_grid = np.column_stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()])

        feature_names = ["x", "y", "z"]

        # Add geological features if requested
        if add_geological_features:
            # Distance from center
            center_x, center_y = nx // 2, ny // 2
            dist_from_center = np.sqrt(
                (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
            ).ravel()

            # Depth-related features
            depth_normalized = z_coords.ravel() / nz

            X_grid = np.column_stack([X_grid, dist_from_center, depth_normalized])
            feature_names.extend(["dist_from_center", "depth_normalized"])

        # Prepare target values
        y_grid = permx_3d.ravel()

        # Apply log transform if requested
        if log_transform:
            # Add small constant to avoid log(0)
            y_grid = np.log(y_grid + 1e-10)

        # Create valid mask (remove invalid values)
        valid_mask = np.isfinite(y_grid) & np.isfinite(X_grid).all(axis=1)

        self.grid_data = GridData(
            X_grid=X_grid[valid_mask],
            y_grid=y_grid[valid_mask],
            feature_names=feature_names,
            permx_3d=permx_3d,
            dimensions=(nx, ny, nz),
            valid_mask=valid_mask,
        )

        logger.info(
            "Features prepared: %d features, %d valid samples",
            len(feature_names),
            len(self.grid_data.X_grid),
        )
        return self.grid_data

    def create_train_test_split(
        self, *, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train/test split.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If features haven't been prepared
        """
        if self.grid_data is None:
            raise ValueError(
                "Features must be prepared first. Call prepare_features()."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            self.grid_data.X_grid,
            self.grid_data.y_grid,
            test_size=test_size,
            random_state=random_state,
        )

        # Store in grid_data
        self.grid_data.X_train = X_train
        self.grid_data.X_test = X_test
        self.grid_data.y_train = y_train
        self.grid_data.y_test = y_test

        logger.info("Train/test split: %d train, %d test samples", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def setup_scalers(
        self, *, scaler_type: str = "standard"
    ) -> tuple[StandardScaler, StandardScaler]:
        """Set up feature and target scalers.

        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')

        Returns:
            Tuple of (x_scaler, y_scaler)

        Raises:
            ValueError: If train/test split hasn't been created
        """
        if self.grid_data is None or self.grid_data.X_train is None:
            raise ValueError(
                "Train/test split must be created first. Call create_train_test_split()."
            )

        if scaler_type == "standard":
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler

            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()

        # Fit scalers
        x_scaler.fit(self.grid_data.X_train)
        y_scaler.fit(self.grid_data.y_train.reshape(-1, 1))

        # Store scaled training data
        self.grid_data.X_train_scaled = x_scaler.transform(self.grid_data.X_train)
        self.grid_data.y_train_scaled = y_scaler.transform(
            self.grid_data.y_train.reshape(-1, 1)
        ).ravel()

        self.scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}

        logger.info("Scalers setup: %s", scaler_type)
        return x_scaler, y_scaler

    def create_model(
        self, model_type: str, *, kernel_type: str = "combined", **kwargs
    ) -> BaseEstimator:
        """Create a model based on the current backend.

        Args:
            model_type: Type of model ('gpr', 'rf', 'svr' for sklearn)
            kernel_type: Kernel type for GP models
            **kwargs: Additional model parameters

        Returns:
            Configured model instance

        Raises:
            ValueError: If backend is gpytorch but model_type is sklearn-specific
        """
        if self.backend == "sklearn":
            return self._create_sklearn_model(
                model_type, kernel_type=kernel_type, **kwargs
            )
        elif self.backend == "gpytorch":
            if model_type not in ["gp", "deep_gp"]:
                raise ValueError(
                    f"GPyTorch backend only supports 'gp' and 'deep_gp' models, got '{model_type}'"
                )
            return self._create_gpytorch_model(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _create_sklearn_model(
        self, model_type: str, *, kernel_type: str = "combined", **kwargs
    ) -> BaseEstimator:
        """Create scikit-learn model."""
        if model_type == "gpr":
            n_features = len(self.grid_data.feature_names) if self.grid_data else 3
            length_scales = [1.0] * n_features

            kernels = {
                "rbf": ConstantKernel(1.0) * RBF(length_scales) + WhiteKernel(1e-3),
                "matern": ConstantKernel(1.0) * Matern(length_scales, nu=1.5)
                + WhiteKernel(1e-3),
                "combined": (
                    ConstantKernel(1.0) * RBF(length_scales)
                    + ConstantKernel(1.0) * Matern(length_scales, nu=1.5)
                    + WhiteKernel(1e-3)
                ),
            }

            kernel = kernels.get(kernel_type, kernels["combined"])
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=kwargs.get("alpha", 1e-6),
                n_restarts_optimizer=kwargs.get("n_restarts_optimizer", 5),
                random_state=kwargs.get("random_state", 42),
            )
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
            )
        elif model_type == "svr":
            model = SVR(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                epsilon=kwargs.get("epsilon", 0.1),
            )
        else:
            raise ValueError(f"Unknown sklearn model type: {model_type}")

        return model

    def _create_gpytorch_model(self, **kwargs) -> tuple[Any, Any]:
        """Create GPyTorch model and likelihood."""
        if not GPYTORCH_AVAILABLE:
            raise ValueError("GPyTorch is not available")

        if self.grid_data is None or self.grid_data.X_train_scaled is None:
            raise ValueError("Scaled training data required for GPyTorch models")

        # Convert to tensors
        X_tensor = torch.tensor(self.grid_data.X_train_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.grid_data.y_train_scaled, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_tensor, y_tensor, likelihood)

        return model, likelihood

    def train_model(
        self, model: BaseEstimator, model_name: str, **kwargs
    ) -> BaseEstimator:
        """Train a model.

        Args:
            model: Model instance to train
            model_name: Name to store the model under
            **kwargs: Additional training parameters

        Returns:
            Trained model

        Raises:
            ValueError: If scalers haven't been set up
        """
        if not self.scalers:
            raise ValueError("Scalers must be set up first. Call setup_scalers().")

        if self.backend == "sklearn":
            logger.info("Training %s model...", model_name)
            model.fit(self.grid_data.X_train_scaled, self.grid_data.y_train_scaled)
            self.models[model_name] = model
            logger.info("DONE: %s training completed", model_name)

        elif self.backend == "gpytorch":
            # GPyTorch training would go here
            logger.warning("GPyTorch training for %s not implemented in this version", model_name)
            self.models[model_name] = model

        return model

    def evaluate_model(
        self, model_name: str, *, return_predictions: bool = False
    ) -> ModelResults:
        """Evaluate a trained model.

        Args:
            model_name: Name of the model to evaluate
            return_predictions: Whether to return predictions

        Returns:
            ModelResults containing evaluation metrics

        Raises:
            ValueError: If model hasn't been trained
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            )

        model = self.models[model_name]
        x_scaler = self.scalers["x_scaler"]
        y_scaler = self.scalers["y_scaler"]

        # Scale test data and make predictions
        X_test_scaled = x_scaler.transform(self.grid_data.X_test)
        y_pred_scaled = model.predict(X_test_scaled)

        # Inverse transform predictions
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(
            self.grid_data.y_test.reshape(-1, 1)
        ).ravel()

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Get uncertainty estimates if available
        y_std = None
        if hasattr(model, "predict") and hasattr(model, "kernel_"):
            try:
                _, y_std_scaled = model.predict(X_test_scaled, return_std=True)
                y_std = y_scaler.scale_ * y_std_scaled
            except:
                pass

        results = ModelResults(r2=r2, rmse=rmse, mae=mae, y_pred=y_pred, y_std=y_std)

        self.results[model_name] = results

        logger.info("RESULTS: %s Results:", model_name)
        logger.info("   R2 Score: %.4f", r2)
        logger.info("   RMSE: %.4f", rmse)
        logger.info("   MAE: %.4f", mae)

        return results

    def save_model(self, model_name: str, filepath: str | Path) -> None:
        """Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            filepath: Path to save the model

        Raises:
            ValueError: If model hasn't been trained
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model_data = {
            "model": self.models[model_name],
            "scalers": self.scalers,
            "feature_names": self.grid_data.feature_names if self.grid_data else None,
            "backend": self.backend,
        }

        joblib.dump(model_data, filepath)
        logger.info("Model '%s' saved to %s", model_name, filepath)

    def load_model(self, model_name: str, filepath: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.models[model_name] = model_data["model"]
        self.scalers = model_data["scalers"]

        logger.info("Model '%s' loaded from %s", model_name, filepath)


def main():
    """Example usage of the unified SPE9 Toolkit."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("SPE9 Unified Geomodeling Toolkit")

    # Initialize toolkit
    toolkit = SPE9Toolkit(backend="sklearn")

    # Load and prepare data
    toolkit.load_data()
    toolkit.prepare_features(add_geological_features=True)
    toolkit.create_train_test_split(test_size=0.2)
    toolkit.setup_scalers()

    # Train models
    logger.info("Training models...")

    # Gaussian Process Regression
    gpr = toolkit.create_model("gpr", kernel_type="combined")
    toolkit.train_model(gpr, "GPR")

    # Random Forest
    rf = toolkit.create_model("rf", n_estimators=50)
    toolkit.train_model(rf, "RandomForest")

    # Evaluate models
    logger.info("Evaluating models...")
    gpr_results = toolkit.evaluate_model("GPR")
    rf_results = toolkit.evaluate_model("RandomForest")

    logger.info(
        "Best model: %s",
        'GPR' if gpr_results.r2 > rf_results.r2 else 'RandomForest'
    )


if __name__ == "__main__":
    main()
