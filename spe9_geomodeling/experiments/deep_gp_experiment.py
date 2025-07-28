"""Deep GP Experiment for SPE9 Reservoir Modeling.

This script compares traditional Gaussian Process models with Deep GP models
to analyze complex spatial patterns in the SPE9 reservoir dataset.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import gpytorch
from tqdm import tqdm

from ..grdecl_parser import GRDECLParser
from model_gp import SPE9GPModel, DeepGPModel, create_gp_model
from plot import SPE9Plotter


class DeepGPExperiment:
    """Experiment framework for comparing traditional GP vs Deep GP models."""

    def __init__(
        self,
        data_path: str = "~/Documents/Pandey_Ch05_Geomodeling_Code/data/SPE9.GRDECL",
        random_state: int = 42,
    ):
        """Initialize the experiment.

        Args:
            data_path: Path to SPE9 GRDECL file
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path).expanduser()
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.scalers = {}

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Initialize plotter
        self.plotter = SPE9Plotter(figsize=(15, 10), dpi=150)

        print("🔬 Deep GP Experiment Initialized")
        print(f"📁 Data path: {self.data_path}")
        print(f"🎲 Random state: {random_state}")

    def load_and_prepare_data(
        self,
        property_name: str = "PERMX",
        train_size: float = 0.7,
        max_samples: Optional[int] = 2000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load SPE9 data and prepare for GP modeling.

        Args:
            property_name: Property to model (default: PERMX)
            train_size: Fraction of data for training
            max_samples: Maximum number of samples to use (for computational efficiency)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as torch tensors
        """
        print(f"\n📊 Loading {property_name} data from SPE9...")

        # Load data
        parser = GRDECLParser(str(self.data_path))
        data = parser.load_data()

        if property_name not in data["properties"]:
            raise ValueError(f"Property {property_name} not found in dataset")

        # Get grid dimensions and property data
        nx, ny, nz = data["dimensions"]
        property_data = data["properties"][property_name]

        print(f"📐 Grid dimensions: {nx} × {ny} × {nz} = {nx*ny*nz} cells")

        # Create coordinate grid
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )

        # Flatten and filter valid data
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()])
        values = property_data.ravel()

        # Remove invalid values
        valid_mask = (values > 0) & np.isfinite(values)
        coords = coords[valid_mask]
        values = values[valid_mask]

        print(f"✅ Valid cells: {len(values)} / {nx*ny*nz}")
        print(f"📈 {property_name} range: {values.min():.2f} - {values.max():.2f}")

        # Subsample if needed for computational efficiency
        if max_samples and len(values) > max_samples:
            indices = np.random.choice(len(values), max_samples, replace=False)
            coords = coords[indices]
            values = values[indices]
            print(f"🎯 Subsampled to {max_samples} points for efficiency")

        # Log-transform the values (common for permeability)
        log_values = np.log10(values + 1e-8)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            coords,
            log_values,
            train_size=train_size,
            random_state=self.random_state,
            stratify=None,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store scaler for later use
        self.scalers["feature_scaler"] = scaler

        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)

        print(f"🚂 Training set: {len(X_train_tensor)} samples")
        print(f"🧪 Test set: {len(X_test_tensor)} samples")

        # Store data for later use
        self.data = {
            "X_train": X_train_tensor,
            "X_test": X_test_tensor,
            "y_train": y_train_tensor,
            "y_test": y_test_tensor,
            "original_coords": coords,
            "original_values": values,
            "log_values": log_values,
            "property_name": property_name,
        }

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

    def train_model(
        self,
        model_name: str,
        model_type: str = "standard",
        training_iter: int = 100,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """Train a GP model.

        Args:
            model_name: Name for this model configuration
            model_type: 'standard' or 'deep'
            training_iter: Number of training iterations
            **model_kwargs: Additional arguments for model creation

        Returns:
            Dictionary with training results
        """
        print(f"\n🚀 Training {model_name} ({model_type} GP)...")

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]

        # Create model and likelihood
        model, likelihood = create_gp_model(
            X_train, y_train, model_type=model_type, **model_kwargs
        )

        # Set to training mode
        model.train()
        likelihood.train()

        # Use Adam optimizer - properly handle shared parameters
        # Get all unique parameters to avoid duplication
        model_params = list(model.parameters())
        likelihood_params = list(likelihood.parameters())

        # Filter out any parameters that might be shared
        all_param_ids = set(id(p) for p in model_params)
        unique_likelihood_params = [
            p for p in likelihood_params if id(p) not in all_param_ids
        ]

        all_params = model_params + unique_likelihood_params
        optimizer = torch.optim.Adam(all_params, lr=0.1)

        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        losses = []
        start_time = time.time()

        with tqdm(range(training_iter), desc=f"Training {model_name}") as pbar:
            for i in pbar:
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        training_time = time.time() - start_time

        # Store model
        self.models[model_name] = {
            "model": model,
            "likelihood": likelihood,
            "type": model_type,
            "training_time": training_time,
            "losses": losses,
            "final_loss": losses[-1],
        }

        print(f"✅ {model_name} trained in {training_time:.2f}s")
        print(f"📉 Final loss: {losses[-1]:.4f}")

        return self.models[model_name]

    def evaluate_model(self, model_name: str) -> Dict[str, float]:
        """Evaluate a trained model.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n📊 Evaluating {model_name}...")

        model_info = self.models[model_name]
        model = model_info["model"]
        likelihood = model_info["likelihood"]

        # Set to evaluation mode
        model.eval()
        likelihood.eval()

        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(X_test))
            y_pred_mean = predictions.mean.numpy()
            y_pred_std = predictions.stddev.numpy()

        # Calculate metrics
        y_test_np = y_test.numpy()

        metrics = {
            "r2_score": r2_score(y_test_np, y_pred_mean),
            "rmse": np.sqrt(mean_squared_error(y_test_np, y_pred_mean)),
            "mae": mean_absolute_error(y_test_np, y_pred_mean),
            "mean_uncertainty": np.mean(y_pred_std),
            "training_time": model_info["training_time"],
            "final_loss": model_info["final_loss"],
        }

        # Store results
        self.results[model_name] = {
            "metrics": metrics,
            "predictions": {"mean": y_pred_mean, "std": y_pred_std, "true": y_test_np},
        }

        print(f"📈 R² Score: {metrics['r2_score']:.4f}")
        print(f"📏 RMSE: {metrics['rmse']:.4f}")
        print(f"📐 MAE: {metrics['mae']:.4f}")
        print(f"🎯 Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")

        return metrics

    def run_comparison_experiment(self) -> Dict[str, Any]:
        """Run comprehensive comparison between different GP models."""
        print("\n🔬 Starting Deep GP Comparison Experiment")
        print("=" * 60)

        # Load data
        self.load_and_prepare_data(max_samples=1500)  # Reasonable size for Deep GP

        # Model configurations to test
        model_configs = {
            "Standard_GP_RBF": {
                "model_type": "standard",
                "kernel_type": "rbf",
                "ard": True,
            },
            "Standard_GP_Matern": {
                "model_type": "standard",
                "kernel_type": "matern",
                "ard": True,
            },
            "Standard_GP_Combined": {
                "model_type": "standard",
                "kernel_type": "combined",
                "ard": True,
            },
            "Deep_GP_Small": {"model_type": "deep", "hidden_dim": 32, "num_layers": 2},
            "Deep_GP_Medium": {"model_type": "deep", "hidden_dim": 64, "num_layers": 3},
            "Deep_GP_Large": {"model_type": "deep", "hidden_dim": 128, "num_layers": 3},
        }

        # Train all models
        for model_name, config in model_configs.items():
            try:
                self.train_model(model_name, training_iter=150, **config)
                self.evaluate_model(model_name)
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        # Generate comparison plots
        self.plot_comparison_results()

        return self.results

    def plot_comparison_results(self):
        """Generate comprehensive comparison plots."""
        print("\n📊 Generating comparison plots...")

        if not self.results:
            print("❌ No results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Deep GP vs Traditional GP Comparison - SPE9 Reservoir", fontsize=16
        )

        # Extract data for plotting
        model_names = list(self.results.keys())
        metrics_data = {name: self.results[name]["metrics"] for name in model_names}

        # 1. R² Score comparison
        ax = axes[0, 0]
        r2_scores = [metrics_data[name]["r2_score"] for name in model_names]
        colors = ["skyblue" if "Deep" not in name else "coral" for name in model_names]
        bars = ax.bar(range(len(model_names)), r2_scores, color=colors)
        ax.set_title("R² Score Comparison")
        ax.set_ylabel("R² Score")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. RMSE comparison
        ax = axes[0, 1]
        rmse_values = [metrics_data[name]["rmse"] for name in model_names]
        bars = ax.bar(range(len(model_names)), rmse_values, color=colors)
        ax.set_title("RMSE Comparison")
        ax.set_ylabel("RMSE")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, rmse in zip(bars, rmse_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rmse:.3f}",
                ha="center",
                va="bottom",
            )

        # 3. Training time comparison
        ax = axes[0, 2]
        train_times = [metrics_data[name]["training_time"] for name in model_names]
        bars = ax.bar(range(len(model_names)), train_times, color=colors)
        ax.set_title("Training Time Comparison")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, time_val in zip(bars, train_times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(train_times) * 0.02,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 4. Prediction vs True scatter plot (best model)
        best_model = max(model_names, key=lambda x: metrics_data[x]["r2_score"])
        ax = axes[1, 0]
        pred_data = self.results[best_model]["predictions"]
        ax.scatter(pred_data["true"], pred_data["mean"], alpha=0.6, s=20)
        ax.plot(
            [pred_data["true"].min(), pred_data["true"].max()],
            [pred_data["true"].min(), pred_data["true"].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("True Values (log10)")
        ax.set_ylabel("Predicted Values (log10)")
        ax.set_title(f"Best Model: {best_model}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 5. Uncertainty vs Error plot
        ax = axes[1, 1]
        errors = np.abs(pred_data["true"] - pred_data["mean"])
        ax.scatter(pred_data["std"], errors, alpha=0.6, s=20)
        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Uncertainty vs Error")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 6. Training loss curves
        ax = axes[1, 2]
        for model_name in model_names:
            if model_name in self.models:
                losses = self.models[model_name]["losses"]
                color = "blue" if "Deep" not in model_name else "red"
                ax.plot(losses, label=model_name, color=color, alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig("deep_gp_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print summary
        print("\n📋 Experiment Summary:")
        print("=" * 50)
        for name in model_names:
            metrics = metrics_data[name]
            model_type = "🧠 Deep GP" if "Deep" in name else "📊 Standard GP"
            print(f"{model_type} {name}:")
            print(
                f"  R²: {metrics['r2_score']:.4f} | RMSE: {metrics['rmse']:.4f} | Time: {metrics['training_time']:.1f}s"
            )

        best_model = max(model_names, key=lambda x: metrics_data[x]["r2_score"])
        print(
            f"\n🏆 Best Model: {best_model} (R² = {metrics_data[best_model]['r2_score']:.4f})"
        )


def main():
    """Run the Deep GP experiment."""
    print("🚀 Starting Deep GP Spatial Pattern Analysis")
    print("=" * 60)

    # Create experiment
    experiment = DeepGPExperiment()

    # Run comparison
    results = experiment.run_comparison_experiment()

    print("\n✅ Experiment completed!")
    print("📁 Results saved to: deep_gp_comparison.png")

    return experiment, results


if __name__ == "__main__":
    experiment, results = main()
