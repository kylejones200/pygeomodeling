"""Unified SPE9 Geomodeling Toolkit.

Supports both scikit-learn and GPyTorch workflows in a single, Pythonic interface.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Import GPyTorch components (optional)
try:
    import torch
    import gpytorch
    from model_gp import GPModel

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    torch = None
    gpytorch = None
    GPModel = None

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
        self, data_path: Optional[Union[str, Path]] = None, backend: str = "sklearn"
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

        self.data: Optional[Dict[str, Any]] = None
        self.X_grid: Optional[np.ndarray] = None
        self.y_grid: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.permx_3d: Optional[np.ndarray] = None
        self.dimensions: Optional[Tuple[int, int, int]] = None

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.valid_mask: Optional[np.ndarray] = None

        # Models and results
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

        print(f"Unified SPE9 Toolkit initialized with {backend} backend")

    def load_data(self) -> Dict[str, Any]:
        """Load SPE9 dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"SPE9 data file not found: {self.data_path}")

        print(f"Loading SPE9 dataset from {self.data_path}")
        self.data = load_spe9_data(str(self.data_path))

        nx, ny, nz = self.data["dimensions"]
        self.permx_3d = self.data["properties"]["PERMX"]
        self.dimensions = (nx, ny, nz)

        print(f"Grid dimensions: {nx} × {ny} × {nz}")
        print(f"PERMX range: {self.permx_3d.min():.2f} - {self.permx_3d.max():.2f} mD")
        print(f"PERMX mean: {self.permx_3d.mean():.2f} mD")

        return self.data

    def prepare_features(
        self, *, add_geological_features: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        print(f"Features prepared: {feature_names}")
        return self.X_grid, self.y_grid

    def create_train_test_split(
        self,
        *,
        test_size: float = 0.2,
        train_size: Optional[int] = None,
        min_perm: float = 1.0,
        random_state: int = 42,
        log_transform: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            print("Applied log1p transform to target values")

        print(f"Valid cells: {len(y_valid):,} out of {len(self.y_grid):,}")

        # Handle train_size parameter for GPyTorch workflow
        if train_size is not None:
            # Sample down for computational efficiency
            if train_size < len(y_valid):
                X_valid, _, y_valid, _ = train_test_split(
                    X_valid, y_valid, train_size=train_size, random_state=random_state
                )
                print(f"Sampled down to {train_size:,} points for efficiency")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state
        )

        print(
            f"Training samples: {len(self.X_train):,}, Test samples: {len(self.X_test):,}"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def setup_scalers(
        self, *, scaler_type: str = "standard"
    ) -> Tuple[StandardScaler, StandardScaler]:
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

        self.scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}

        print(f"Scalers setup: {scaler_type}")
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

    def create_gpytorch_model(self, **kwargs) -> Tuple[Any, Any]:
        """Create GPyTorch model and likelihood."""
        if self.backend != "gpytorch":
            raise ValueError("GPyTorch models require 'gpytorch' backend")

        if not GPYTORCH_AVAILABLE:
            raise ValueError("GPyTorch is not available")

        # Scale training data
        X_scaled = self.scalers["x_scaler"].transform(self.X_train)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_tensor, y_tensor, likelihood)

        return model, likelihood

    def train_sklearn_model(
        self, model: BaseEstimator, model_name: str
    ) -> BaseEstimator:
        """Train scikit-learn model."""
        X_scaled = self.scalers["x_scaler"].transform(self.X_train)
        y_scaled = (
            self.scalers["y_scaler"].transform(self.y_train.reshape(-1, 1)).flatten()
        )

        print(f"Training {model_name} (sklearn)...")
        model.fit(X_scaled, y_scaled)

        self.models[model_name] = model
        print(f"{model_name} trained successfully!")

        if hasattr(model, "kernel_"):
            print(f"Final kernel: {model.kernel_}")

        return model

    def train_gpytorch_model(
        self,
        model: Any,
        likelihood: Any,
        model_name: str,
        *,
        n_iter: int = 100,
        lr: float = 0.1,
    ) -> Tuple[Any, Any]:
        """Train GPyTorch model."""
        if not GPYTORCH_AVAILABLE:
            raise ValueError("GPyTorch is not available")

        X_scaled = self.scalers["x_scaler"].transform(self.X_train)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        print(f"Training {model_name} (GPyTorch)...")
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()

            if i % 20 == 0:
                print(f"  Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}")

            optimizer.step()

        self.models[model_name] = {"model": model, "likelihood": likelihood}
        print(f"{model_name} trained successfully!")

        return model, likelihood

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")

        model = self.models[model_name]
        X_test_scaled = self.scalers["x_scaler"].transform(self.X_test)

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

        print(f"{model_name} Results:")
        print(f"  R²: {results['r2']:.3f}")
        print(f"  RMSE: {results['rmse']:.2f}")
        print(f"  MAE: {results['mae']:.2f}")

        return results

    def save_model(self, model_name: str, output_dir: Optional[Path] = None) -> None:
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

        print(f"Model {model_name} saved to {output_dir}")

    def visualize_results(
        self,
        model_name: str,
        *,
        z_slice: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 10),
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
        plt.colorbar(im1, ax=axes[0, 0], label="mD")

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

        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        axes[1, 0].set_xlabel("True Values")
        axes[1, 0].set_ylabel("Predicted Values")
        axes[1, 0].set_title(f"{model_name}: Predicted vs Actual")

        # Residuals
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color="r", linestyle="--")
        axes[1, 1].set_xlabel("Predicted Values")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals vs Predicted")

        plt.tight_layout()
        filename = f"{model_name.lower()}_{self.backend}_results.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Visualization saved: {filename}")
        plt.show()


def main() -> None:
    """Example usage of the Unified SPE9 Toolkit."""
    print("Unified SPE9 Geomodeling Toolkit")
    print("=" * 50)
    print("Supports both scikit-learn and GPyTorch backends")
    print("\nExample usage:")
    print("# Scikit-learn workflow")
    print("toolkit = UnifiedSPE9Toolkit(backend='sklearn')")
    print("toolkit.load_data()")
    print("toolkit.prepare_features()")
    print("toolkit.create_train_test_split()")
    print("toolkit.setup_scalers()")
    print("gpr = toolkit.create_sklearn_model('gpr')")
    print("toolkit.train_sklearn_model(gpr, 'GPR')")
    print("toolkit.evaluate_model('GPR')")
    print()
    print("# GPyTorch workflow")
    print("toolkit = UnifiedSPE9Toolkit(backend='gpytorch')")
    print("toolkit.load_data()")
    print("toolkit.prepare_features()")
    print("toolkit.create_train_test_split(train_size=3000, log_transform=True)")
    print("toolkit.setup_scalers()")
    print("model, likelihood = toolkit.create_gpytorch_model()")
    print("toolkit.train_gpytorch_model(model, likelihood, 'GPyTorch_GP')")
    print("toolkit.evaluate_model('GPyTorch_GP')")


if __name__ == "__main__":
    main()
