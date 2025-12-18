"""Unified Geomodeling Toolkit.

Supports both scikit-learn and GPyTorch workflows in a single, Pythonic interface.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
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
except ImportError as exc:
    GPYTORCH_AVAILABLE = False
    torch = None
    gpytorch = None
    GPModel = None  # type: ignore[assignment]
    warnings.warn(
        "GPyTorch backend is unavailable. Install the 'advanced' extras to enable it.",
        RuntimeWarning,
    )

from .grdecl_parser import load_spe9_data

warnings.filterwarnings("ignore")


class UnifiedSPE9Toolkit:
    """Unified toolkit supporting both scikit-learn and GPyTorch workflows.

    This toolkit provides a consistent interface for geomodeling with the SPE9 dataset,
    supporting both traditional scikit-learn models and advanced GPyTorch models.

    Attributes:
        data_path: Path to SPE9 dataset file
        backend: Modeling backend ('sklearn' or 'gpytorch')
        data: Loaded SPE9 dataset
        models: Dictionary of trained models
        scalers: Dictionary of data scalers
        results: Dictionary of model evaluation results
    """

    def __init__(
        self, data_path: str | Path | None = None, backend: str = "sklearn"
    ) -> None:
        """Initialize the unified toolkit.

        Args:
            data_path: Path to SPE9 dataset file
            backend: Modeling backend ('sklearn' or 'gpytorch')

        Raises:
            ValueError: If backend is invalid or GPyTorch backend is requested but not available
        """
        if backend not in ["sklearn", "gpytorch"]:
            raise ValueError("Backend must be 'sklearn' or 'gpytorch'")

        if backend == "gpytorch" and not GPYTORCH_AVAILABLE:
            raise ValueError(
                "GPyTorch backend requested but GPyTorch is not installed. "
                "Install with: pip install torch gpytorch"
            )

        if data_path is None:
            # Use the bundled data file in the project
            module_dir = Path(__file__).parent.parent
            default_path = module_dir / "data" / "SPE9.GRDECL"
        else:
            default_path = Path(data_path)
        self.data_path = default_path
        self.backend = backend

        self.data: dict[str, Any] | None = None
        self.X_grid: np.ndarray | None = None
        self.y_grid: np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.permx_3d: np.ndarray | None = None
        self.dimensions: tuple[int, int, int] | None = None

        # Training data
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.X_train_scaled: np.ndarray | None = None
        self.y_train_scaled: np.ndarray | None = None
        self.X_test_scaled: np.ndarray | None = None

        # Models and results
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.X_train_scaled: np.ndarray | None = None
        self.y_train_scaled: np.ndarray | None = None

        logger.info("Unified SPE9 Toolkit initialized with %s backend", backend)

    def load_data(self) -> dict[str, Any]:
        """Load SPE9 dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"SPE9 data file not found: {self.data_path}")

        logger.info("Loading SPE9 dataset from %s", self.data_path)
        self.data = load_spe9_data(str(self.data_path))

        nx, ny, nz = self.data["dimensions"]
        self.permx_3d = self.data["properties"]["PERMX"]
        self.dimensions = (nx, ny, nz)

        logger.info("Grid dimensions: %d x %d x %d", nx, ny, nz)
        logger.info(
            "PERMX range: %.2f - %.2f mD", self.permx_3d.min(), self.permx_3d.max()
        )
        logger.info("PERMX mean: %.2f mD", self.permx_3d.mean())

        return self.data

    def prepare_features(
        self, *, add_geological_features: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare coordinate and geological features."""
        if self.data is None:
            raise ValueError("Load data first using load_data()")

        nx, ny, nz = self.dimensions

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
                dist_center.ravel(),
                Z_full.ravel(),
                (X_full * Y_full).ravel(),
                (X_full * Z_full).ravel(),
                (Y_full * Z_full).ravel(),
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

        self.X_grid = np.column_stack(features)
        self.y_grid = self.permx_3d.ravel()
        self.feature_names = feature_names

        logger.info("Features prepared: %s", feature_names)
        return self.X_grid, self.y_grid

    def create_train_test_split(
        self,
        *,
        test_size: float = 0.2,
        train_size: int | None = None,
        min_perm: float = 1.0,
        random_state: int = 42,
        log_transform: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training and test sets."""
        if self.X_grid is None:
            raise ValueError("Prepare features first using prepare_features()")

        # Filter valid cells
        self.valid_mask = self.y_grid > min_perm
        X_valid = self.X_grid[self.valid_mask]
        y_valid = self.y_grid[self.valid_mask]

        # Apply log transform if requested (useful for GPyTorch)
        if log_transform:
            y_valid = np.log1p(y_valid)
            logger.info("Applied log1p transform to target values")

        logger.info("Valid cells: %d out of %d", len(y_valid), len(self.y_grid))

        # Handle train_size parameter for GPyTorch workflow
        if train_size is not None:
            # Sample down for computational efficiency
            if train_size < len(y_valid):
                X_valid, _, y_valid, _ = train_test_split(
                    X_valid, y_valid, train_size=train_size, random_state=random_state
                )
                logger.info("Sampled down to %d points for efficiency", train_size)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state
        )

        # Invalidate previously scaled data and scalers
        self.scalers.clear()
        self.X_train_scaled = None
        self.y_train_scaled = None
        self.X_test_scaled = None

        logger.info(
            "Training samples: %d, Test samples: %d",
            len(self.X_train),
            len(self.X_test),
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def setup_scalers(
        self, *, scaler_type: str = "standard"
    ) -> tuple[StandardScaler, StandardScaler]:
        """Setup and fit data scalers."""
        if self.X_train is None:
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

        # Fit scalers
        x_scaler.fit(self.X_train)
        y_scaler.fit(self.y_train.reshape(-1, 1))

        # Transform and store scaled training data for downstream workflows
        self.X_train_scaled = x_scaler.transform(self.X_train)
        self.y_train_scaled = y_scaler.transform(self.y_train.reshape(-1, 1)).flatten()

        self.scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}
        self.X_train_scaled = x_scaler.transform(self.X_train)
        self.y_train_scaled = y_scaler.transform(self.y_train.reshape(-1, 1)).ravel()
        self.X_test_scaled = (
            x_scaler.transform(self.X_test) if self.X_test is not None else None
        )

        logger.info("Scalers setup: %s", scaler_type)
        return x_scaler, y_scaler

    def create_sklearn_model(
        self, model_type: str, *, kernel_type: str = "combined", **kwargs
    ) -> BaseEstimator:
        """Create scikit-learn model."""
        if model_type == "gpr":
            n_features = len(self.feature_names) if self.feature_names else 3
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

    def create_gpytorch_model(self, **kwargs) -> tuple[Any, Any]:
        """Create GPyTorch model and likelihood."""
        if self.backend != "gpytorch":
            raise ValueError("GPyTorch models require 'gpytorch' backend")

        if not GPYTORCH_AVAILABLE:
            raise ValueError("GPyTorch is not available")

        # Scale training data
        if self.X_train_scaled is None or self.y_train is None:
            raise ValueError("Call setup_scalers() before creating GPyTorch models.")

        X_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_tensor, y_tensor, likelihood)

        return model, likelihood

    def train_sklearn_model(
        self, model: BaseEstimator, model_name: str
    ) -> BaseEstimator:
        """Train scikit-learn model."""
        if self.X_train_scaled is None or self.y_train_scaled is None:
            raise ValueError("Call setup_scalers() before training models.")

        X_scaled = self.X_train_scaled
        y_scaled = self.y_train_scaled

        logger.info("Training %s (sklearn)...", model_name)
        model.fit(X_scaled, y_scaled)

        self.models[model_name] = model
        logger.info("%s trained successfully!", model_name)

        if hasattr(model, "kernel_"):
            logger.info("Final kernel: %s", model.kernel_)

        return model

    def train_gpytorch_model(
        self,
        model: Any,
        likelihood: Any,
        model_name: str,
        *,
        n_iter: int = 100,
        lr: float = 0.1,
    ) -> tuple[Any, Any]:
        """Train GPyTorch model."""
        if not GPYTORCH_AVAILABLE:
            raise ValueError("GPyTorch is not available")

        if self.X_train_scaled is None or self.y_train is None:
            raise ValueError("Call setup_scalers() before training GPyTorch models.")

        X_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        logger.info("Training %s (GPyTorch)...", model_name)
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()

            if i % 20 == 0:
                logger.info("  Iter %d/%d - Loss: %.3f", i + 1, n_iter, loss.item())

            optimizer.step()

        self.models[model_name] = {"model": model, "likelihood": likelihood}
        logger.info("%s trained successfully!", model_name)

        return model, likelihood

    def evaluate_model(self, model_name: str) -> dict[str, Any]:
        """Evaluate a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")

        model = self.models[model_name]
        if self.X_test is None:
            raise ValueError("Test split not created. Call create_train_test_split().")

        if self.X_test_scaled is None:
            X_test_scaled = self.scalers["x_scaler"].transform(self.X_test)
        else:
            X_test_scaled = self.X_test_scaled

        if self.backend == "sklearn":
            # Scikit-learn model
            if hasattr(model, "predict") and hasattr(model, "kernel_"):  # GPR
                y_pred_scaled, y_std_scaled = model.predict(
                    X_test_scaled, return_std=True
                )
                y_std = y_std_scaled * self.scalers["y_scaler"].scale_[0]
            else:  # Other models
                y_pred_scaled = model.predict(X_test_scaled)
                y_std = None

            y_pred = (
                self.scalers["y_scaler"]
                .inverse_transform(y_pred_scaled.reshape(-1, 1))
                .flatten()
            )

        else:
            # GPyTorch model
            gp_model = model["model"]
            likelihood = model["likelihood"]

            gp_model.eval()
            likelihood.eval()

            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = likelihood(gp_model(X_tensor))
                y_pred = preds.mean.numpy()
                y_std = preds.stddev.numpy()

        # Calculate metrics
        results = {
            "r2": r2_score(self.y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
            "mae": mean_absolute_error(self.y_test, y_pred),
            "y_pred": y_pred,
            "y_std": y_std,
        }

        self.results[model_name] = results

        logger.info("%s Results:", model_name)
        logger.info("  R2: %.3f", results["r2"])
        logger.info("  RMSE: %.2f", results["rmse"])
        logger.info("  MAE: %.2f", results["mae"])

        return results

    def save_model(self, model_name: str, output_dir: Path | None = None) -> None:
        """Save trained model and scalers."""
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        if self.backend == "sklearn":
            # Save sklearn model
            joblib.dump(model, output_dir / f"{model_name}_sklearn.joblib")
        else:
            # Save GPyTorch model
            torch.save(
                model["model"].state_dict(), output_dir / f"{model_name}_model.pth"
            )
            torch.save(
                model["likelihood"].state_dict(),
                output_dir / f"{model_name}_likelihood.pth",
            )

        # Save scalers
        joblib.dump(self.scalers, output_dir / f"{model_name}_scalers.joblib")

        logger.info("Model %s saved to %s", model_name, output_dir)

    def visualize_results(
        self,
        model_name: str,
        *,
        z_slice: int | None = None,
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """Create visualizations for a model."""
        if model_name not in self.results:
            raise ValueError(f"Evaluate {model_name} first")

        if z_slice is None:
            z_slice = self.dimensions[2] // 2

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Original PERMX
        im1 = axes[0, 0].imshow(
            self.permx_3d[:, :, z_slice].T, origin="lower", cmap="viridis"
        )
        axes[0, 0].set_title(f"Original PERMX (Z={z_slice})")
        plt.colorbar(im1, ax=axes[0, 0], label="mD", shrink=0.8)

        # Model comparison (placeholder for now)
        axes[0, 1].text(
            0.5,
            0.5,
            f"{model_name}\n({self.backend})\nResults",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=14,
        )
        axes[0, 1].set_title(f"{model_name} Model Info")

        # Predictions vs actual
        y_test = self.y_test
        y_pred = self.results[model_name]["y_pred"]

        axes[1, 0].scatter(y_test, y_pred, alpha=0.6, color="#555555")
        axes[1, 0].plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color=signalplot.ACCENT,
            ls="--",
            lw=2,
        )
        axes[1, 0].set_xlabel("True Values")
        axes[1, 0].set_ylabel("Predicted Values")
        axes[1, 0].set_title(f"Predicted vs Actual")

        # Residuals
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color="#555555")
        axes[1, 1].axhline(y=0, color=signalplot.ACCENT, linestyle="--")
        axes[1, 1].set_xlabel("Predicted Values")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals vs Predicted")

        plt.tight_layout()
        filename = f"{model_name.lower()}_{self.backend}_results.png"
        plt.savefig(filename)
        logger.info("Visualization saved: %s", filename)
        plt.show()


def main() -> None:
    """Example usage of the Unified SPE9 Toolkit."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Unified SPE9 Geomodeling Toolkit")
    logger.info("Supports both scikit-learn and GPyTorch backends")
    logger.info("\nExample usage:")
    logger.info("# Scikit-learn workflow")
    logger.info("toolkit = UnifiedSPE9Toolkit(backend='sklearn')")
    logger.info("toolkit.load_data()")
    logger.info("toolkit.prepare_features()")
    logger.info("toolkit.create_train_test_split()")
    logger.info("toolkit.setup_scalers()")
    logger.info("gpr = toolkit.create_sklearn_model('gpr')")
    logger.info("toolkit.train_sklearn_model(gpr, 'GPR')")
    logger.info("toolkit.evaluate_model('GPR')")
    logger.info("")
    logger.info("# GPyTorch workflow")
    logger.info("toolkit = UnifiedSPE9Toolkit(backend='gpytorch')")
    logger.info("toolkit.load_data()")
    logger.info("toolkit.prepare_features()")
    logger.info("toolkit.create_train_test_split(train_size=3000, log_transform=True)")
    logger.info("toolkit.setup_scalers()")
    logger.info("model, likelihood = toolkit.create_gpytorch_model()")
    logger.info("toolkit.train_gpytorch_model(model, likelihood, 'GPyTorch_GP')")
    logger.info("toolkit.evaluate_model('GPyTorch_GP')")


if __name__ == "__main__":
    main()
