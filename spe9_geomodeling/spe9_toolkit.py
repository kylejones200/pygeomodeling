"""SPE9 Geomodeling Toolkit.

A Pythonic, modular toolkit for reservoir property modeling using the SPE9 dataset.
Follows PEP 8 conventions and Python best practices.

.. deprecated:: 0.3.0
    This module is deprecated. Use :class:`UnifiedSPE9Toolkit` from
    :mod:`spe9_geomodeling.unified_toolkit` instead.
"""

from __future__ import annotations

import logging
import warnings

warnings.warn(
    "spe9_geomodeling.spe9_toolkit.SPE9Toolkit is deprecated. "
    "Use spe9_geomodeling.UnifiedSPE9Toolkit instead.",
    DeprecationWarning,
    stacklevel=2,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import signalplot

# Configure logging
logger = logging.getLogger(__name__)

# Apply signalplot style
signalplot.apply()
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    Matern,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    """Pythonic toolkit for SPE9 geomodeling.

    A modular toolkit that provides explicit control over each step of the
    geomodeling workflow. No automatic training - all operations require
    explicit user action.

    Attributes:
        data_path: Path to SPE9 dataset file
        data: Loaded SPE9 dataset
        grid_data: Container for grid data and features
        models: Dictionary of trained models
        scalers: Dictionary of data scalers
        results: Dictionary of model evaluation results
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        """Initialize the toolkit.

        Args:
            data_path: Path to SPE9 dataset file. If None, uses default path.
        """
        if data_path is None:
            # Use the bundled data file in the project
            module_dir = Path(__file__).parent.parent
            default_path = module_dir / "data" / "SPE9.GRDECL"
        else:
            default_path = Path(data_path)
        self.data_path = default_path

        self.data: dict[str, Any] | None = None
        self.grid_data: GridData | None = None
        self.models: dict[str, BaseEstimator] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.results: dict[str, ModelResults] = {}

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

    def prepare_features(self, *, add_geological_features: bool = False) -> GridData:
        """Prepare coordinate and geological features.

        Args:
            add_geological_features: Whether to add geological interaction features

        Returns:
            GridData object containing features and target values

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Load data first using load_data()")

        nx, ny, nz = self.data["dimensions"]
        permx_3d = self.data["properties"]["PERMX"]

        # Create normalized coordinate grids
        x_coords = np.linspace(0, 1, nx)
        y_coords = np.linspace(0, 1, ny)
        z_coords = np.linspace(0, 1, nz)
        X_full, Y_full, Z_full = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )

        # Basic coordinate features
        features = [X_full.ravel(), Y_full.ravel(), Z_full.ravel()]
        feature_names = ["x", "y", "z"]

        if add_geological_features:
            # Add geological context features
            center_x, center_y = 0.5, 0.5
            dist_center = np.sqrt((X_full - center_x) ** 2 + (Y_full - center_y) ** 2)

            additional_features = [
                dist_center.ravel(),  # Distance from center
                Z_full.ravel(),  # Depth factor
                (X_full * Y_full).ravel(),  # XY interaction
                (X_full * Z_full).ravel(),  # XZ interaction
                (Y_full * Z_full).ravel(),  # YZ interaction
            ]

            features.extend(additional_features)
            feature_names.extend(
                [
                    "dist_center",
                    "depth_factor",
                    "xy_interaction",
                    "xz_interaction",
                    "yz_interaction",
                ]
            )

        X_grid = np.column_stack(features)
        y_grid = permx_3d.ravel()

        self.grid_data = GridData(
            X_grid=X_grid,
            y_grid=y_grid,
            feature_names=feature_names,
            permx_3d=permx_3d,
            dimensions=(nx, ny, nz),
        )

        logger.info("Features prepared: %s", feature_names)
        return self.grid_data

    def create_train_test_split(
        self, *, test_size: float = 0.2, min_perm: float = 1.0, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training and test sets.

        Args:
            test_size: Fraction of data to use for testing
            min_perm: Minimum permeability threshold for valid cells
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If features haven't been prepared yet
        """
        if self.grid_data is None:
            raise ValueError("Prepare features first using prepare_features()")

        # Filter valid cells (remove very low permeability)
        valid_mask = self.grid_data.y_grid > min_perm
        X_valid = self.grid_data.X_grid[valid_mask]
        y_valid = self.grid_data.y_grid[valid_mask]

        logger.info("Valid cells: %d out of %d", len(y_valid), len(self.grid_data.y_grid))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state
        )

        logger.info("Training samples: %d, Test samples: %d", len(X_train), len(X_test))

        # Update grid_data
        self.grid_data.X_train = X_train
        self.grid_data.X_test = X_test
        self.grid_data.y_train = y_train
        self.grid_data.y_test = y_test
        self.grid_data.valid_mask = valid_mask

        return X_train, X_test, y_train, y_test

    def setup_scalers(
        self, *, scaler_type: str = "standard"
    ) -> tuple[StandardScaler, StandardScaler]:
        """Setup and fit data scalers.

        Args:
            scaler_type: Type of scaler ('standard' or 'robust')

        Returns:
            Tuple of (x_scaler, y_scaler)

        Raises:
            ValueError: If train/test split hasn't been created yet
        """
        if self.grid_data is None or self.grid_data.X_train is None:
            raise ValueError("Create train/test split first")

        if scaler_type == "standard":
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
        elif scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler

            x_scaler = RobustScaler()
            y_scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")

        # Fit and transform training data
        self.grid_data.X_train_scaled = x_scaler.fit_transform(
            self.grid_data.X_train
        )
        self.grid_data.y_train_scaled = y_scaler.fit_transform(
            self.grid_data.y_train.reshape(-1, 1)
        ).flatten()

        self.scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}

        logger.info("Scalers setup: %s", scaler_type)
        return x_scaler, y_scaler

    @staticmethod
    def create_gpr_kernel(kernel_type: str = "combined", n_features: int = 3) -> Kernel:
        """Create GPR kernel.

        Args:
            kernel_type: Type of kernel ('rbf', 'matern', 'combined')
            n_features: Number of input features

        Returns:
            Configured kernel object
        """
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

        if kernel_type not in kernels:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        return kernels[kernel_type]

    def create_model(
        self, model_type: str, *, kernel_type: str = "combined", **kwargs
    ) -> BaseEstimator:
        """Create (but don't train) a model.

        Args:
            model_type: Type of model ('gpr' or 'rf')
            kernel_type: For GPR, type of kernel to use
            **kwargs: Additional model parameters

        Returns:
            Configured model object
        """
        if model_type == "gpr":
            n_features = len(self.grid_data.feature_names) if self.grid_data else 3
            kernel = self.create_gpr_kernel(kernel_type, n_features)

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
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info("%s model created", model_type.upper())
        return model

    def train_model(self, model: BaseEstimator, model_name: str) -> BaseEstimator:
        """Train a model explicitly.

        Args:
            model: Model to train
            model_name: Name to store the model under

        Returns:
            Trained model

        Raises:
            ValueError: If scalers haven't been setup yet
        """
        if not self.scalers or self.grid_data.X_train_scaled is None:
            raise ValueError("Setup scalers first")

        logger.info("Training %s...", model_name)
        model.fit(self.grid_data.X_train_scaled, self.grid_data.y_train_scaled)

        self.models[model_name] = model
        logger.info("%s trained successfully!", model_name)

        if hasattr(model, "kernel_"):
            logger.info("Final kernel: %s", model.kernel_)

        return model

    def evaluate_model(self, model_name: str) -> ModelResults:
        """Evaluate a trained model.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            ModelResults object containing evaluation metrics

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")

        model = self.models[model_name]
        X_test_scaled = self.scalers["x_scaler"].transform(self.grid_data.X_test)

        # Make predictions
        if hasattr(model, "predict") and hasattr(model, "kernel_"):  # GPR
            y_pred_scaled, y_std_scaled = model.predict(X_test_scaled, return_std=True)
            y_std = y_std_scaled * self.scalers["y_scaler"].scale_[0]
        else:  # Other models
            y_pred_scaled = model.predict(X_test_scaled)
            y_std = None

        y_pred = (
            self.scalers["y_scaler"]
            .inverse_transform(y_pred_scaled.reshape(-1, 1))
            .flatten()
        )

        # Calculate metrics
        y_test = self.grid_data.y_test
        results = ModelResults(
            r2=r2_score(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mae=mean_absolute_error(y_test, y_pred),
            y_pred=y_pred,
            y_std=y_std,
        )

        self.results[model_name] = results

        logger.info("%s Results:", model_name)
        logger.info("  R2: %.3f", results.r2)
        logger.info("  RMSE: %.2f mD", results.rmse)
        logger.info("  MAE: %.2f mD", results.mae)

        return results

    def visualize_results(
        self,
        model_name: str,
        *,
        z_slice: int | None = None,
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """Create visualizations for a model.

        Args:
            model_name: Name of the model to visualize
            z_slice: Z-slice index for visualization (default: middle slice)
            figsize: Figure size tuple
        """
        if model_name not in self.results:
            raise ValueError(f"Evaluate {model_name} first")

        if z_slice is None:
            z_slice = self.grid_data.dimensions[2] // 2

        # Get full grid predictions
        pred_3d, sigma_3d = self._predict_full_grid(model_name)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Original PERMX
        im1 = axes[0, 0].imshow(
            self.grid_data.permx_3d[:, :, z_slice].T, origin="lower", cmap="viridis"
        )
        axes[0, 0].set_title(f"Original PERMX (Z={z_slice})")
        plt.colorbar(im1, ax=axes[0, 0], label="mD", shrink=0.8)

        # Predicted PERMX
        im2 = axes[0, 1].imshow(
            pred_3d[:, :, z_slice].T, origin="lower", cmap="viridis"
        )
        axes[0, 1].set_title(f"Predicted PERMX (Z={z_slice})")
        plt.colorbar(im2, ax=axes[0, 1], label="mD", shrink=0.8)

        # Uncertainty (if available)
        if self.results[model_name].y_std is not None:
            im3 = axes[1, 0].imshow(
                sigma_3d[:, :, z_slice].T, origin="lower", cmap="plasma"
            )
            axes[1, 0].set_title(f"Uncertainty (Z={z_slice})")
            plt.colorbar(im3, ax=axes[1, 0], label="sigma", shrink=0.8)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No uncertainty\navailable",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Uncertainty Not Available")

        # Predictions vs actual
        y_test = self.grid_data.y_test
        y_pred = self.results[model_name].y_pred

        axes[1, 1].scatter(y_test, y_pred, alpha=0.6, color="#555555")
        axes[1, 1].plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            color=signalplot.ACCENT, ls="--", lw=2
        )
        axes[1, 1].set_xlabel("True PERMX (mD)")
        axes[1, 1].set_ylabel("Predicted PERMX (mD)")
        axes[1, 1].set_title(f"Predicted vs Actual")

        plt.tight_layout()
        filename = f"{model_name.lower()}_results.png"
        plt.savefig(filename)
        logger.info("Visualization saved: %s", filename)
        plt.show()

    def _predict_full_grid(self, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions for the full grid (internal method)."""
        model = self.models[model_name]
        X_grid_scaled = self.scalers["x_scaler"].transform(self.grid_data.X_grid)

        if hasattr(model, "predict") and hasattr(model, "kernel_"):  # GPR
            pred_scaled, sigma_scaled = model.predict(X_grid_scaled, return_std=True)
        else:
            pred_scaled = model.predict(X_grid_scaled)
            sigma_scaled = np.zeros_like(pred_scaled)

        pred_orig = (
            self.scalers["y_scaler"]
            .inverse_transform(pred_scaled.reshape(-1, 1))
            .flatten()
        )

        # Create 3D arrays
        nx, ny, nz = self.grid_data.dimensions
        pred_3d = np.zeros((nx, ny, nz))
        sigma_3d = np.zeros((nx, ny, nz))

        pred_3d_flat = pred_3d.ravel()
        sigma_3d_flat = sigma_3d.ravel()

        pred_3d_flat[self.grid_data.valid_mask] = pred_orig[self.grid_data.valid_mask]
        if hasattr(model, "kernel_"):
            sigma_3d_flat[self.grid_data.valid_mask] = sigma_scaled[
                self.grid_data.valid_mask
            ]

        return pred_3d_flat.reshape((nx, ny, nz)), sigma_3d_flat.reshape((nx, ny, nz))

    def export_to_grdecl(
        self, model_name: str, output_dir: Path | None = None
    ) -> tuple[Path, Path]:
        """Export predictions to GRDECL format.

        Args:
            model_name: Name of the model to export
            output_dir: Directory to save files (default: current directory)

        Returns:
            Tuple of (prediction_file_path, uncertainty_file_path)
        """
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        pred_3d, sigma_3d = self._predict_full_grid(model_name)

        def write_grdecl(
            property_3d: np.ndarray, property_name: str, filepath: Path
        ) -> None:
            """Write 3D property to GRDECL format."""
            with filepath.open("w") as f:
                f.write(f"{property_name}\n")
                values = property_3d.ravel(order="F")

                for i in range(0, len(values), 5):
                    row_values = values[i : i + 5]
                    f.write("  ".join([f"{val:12.5f}" for val in row_values]) + "\n")
                f.write("/\n\n")

        pred_file = output_dir / f"PERMX_{model_name.upper()}.GRDECL"
        sigma_file = output_dir / f"SIGMA_{model_name.upper()}.GRDECL"

        write_grdecl(pred_3d, f"PERMX_{model_name.upper()}", pred_file)
        write_grdecl(sigma_3d, f"SIGMA_{model_name.upper()}", sigma_file)

        logger.info("Exported: %s, %s", pred_file.name, sigma_file.name)
        return pred_file, sigma_file


def main() -> None:
    """Example usage of the SPE9 Toolkit."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("SPE9 Geomodeling Toolkit - Pythonic Version")
    logger.info("No automatic training. Use the toolkit methods explicitly.")
    logger.info("\nExample workflow:")
    logger.info("toolkit = SPE9Toolkit()")
    logger.info("toolkit.load_data()")
    logger.info("toolkit.prepare_features(add_geological_features=True)")
    logger.info("toolkit.create_train_test_split()")
    logger.info("toolkit.setup_scalers()")
    logger.info("gpr = toolkit.create_model('gpr')")
    logger.info("toolkit.train_model(gpr, 'GPR')")
    logger.info("toolkit.evaluate_model('GPR')")
    logger.info("toolkit.visualize_results('GPR')")


if __name__ == "__main__":
    main()
