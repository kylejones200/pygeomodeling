"""Clean plotting utilities for geomodeling.

Focused plotting module with proper separation of concerns.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use("default")
sns.set_palette("husl")


class SPE9Plotter:
    """Clean plotting utilities for SPE9 geomodeling results."""

    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 150):
        """Initialize plotter with default settings.

        Args:
            figsize: Default figure size
            dpi: Default DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 8)

    def plot_slice(
        self,
        data: np.ndarray,
        title: str,
        *,
        filename: str | Path | None = None,
        cmap: str = "viridis",
        log_scale: bool = False,
        colorbar_label: str = "Value",
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """Plot a 2D slice of 3D data.

        Args:
            data: 2D array to plot
            title: Plot title
            filename: Optional filename to save plot
            cmap: Colormap name
            log_scale: Whether to use log scale for colors
            colorbar_label: Label for colorbar
            figsize: Figure size override
        """
        fig_size = figsize or (8, 6)
        fig, ax = plt.subplots(figsize=fig_size)

        norm = LogNorm() if log_scale else None
        im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        plt.show()

    def plot_permeability_slice(
        self,
        permx_3d: np.ndarray,
        z_slice: int = 0,
        *,
        filename: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a permeability slice from 3D permeability data.

        Args:
            permx_3d: 3D permeability array (nx, ny, nz)
            z_slice: Z-index of slice to plot
            filename: Optional filename to save plot
            figsize: Figure size override

        Returns:
            Tuple of (figure, axes) objects
        """
        if permx_3d.ndim != 3:
            raise ValueError(f"Expected 3D array, got {permx_3d.ndim}D")

        # Handle negative indexing (Python style)
        if z_slice < 0:
            z_slice = permx_3d.shape[2] + z_slice

        if z_slice < 0 or z_slice >= permx_3d.shape[2]:
            raise ValueError(f"z_slice out of range [0, {permx_3d.shape[2]-1}]")

        # Extract the 2D slice
        slice_data = permx_3d[:, :, z_slice]

        # Create the plot
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size)

        # Use log scale for permeability
        im = ax.imshow(slice_data.T, origin="lower", cmap=cmap, norm=LogNorm())

        plot_title = title if title is not None else f"Permeability Slice (z={z_slice})"
        ax.set_title(plot_title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Permeability (mD)", rotation=270, labelpad=20)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        return fig, ax

    def plot_prediction_comparison(
        self,
        true_3d: np.ndarray,
        pred_3d: np.ndarray,
        z_slice: int = 0,
        *,
        sigma_3d: np.ndarray | None = None,
        filename: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot comparison between true and predicted values.

        Args:
            true_3d: True 3D data
            pred_3d: Predicted 3D data
            z_slice: Z-index of slice to plot
            sigma_3d: Optional uncertainty data
            filename: Optional filename to save plot
            figsize: Figure size override

        Returns:
            Tuple of (figure, axes array)
        """
        if true_3d.shape != pred_3d.shape:
            raise ValueError("true_3d and pred_3d must have the same shape")

        if true_3d.ndim != 3:
            raise ValueError(f"Expected 3D arrays, got {true_3d.ndim}D")

        # Handle negative indexing
        if z_slice < 0:
            z_slice = true_3d.shape[2] + z_slice

        if z_slice < 0 or z_slice >= true_3d.shape[2]:
            raise ValueError(f"z_slice out of range [0, {true_3d.shape[2]-1}]")

        # Extract slices
        true_slice = true_3d[:, :, z_slice]
        pred_slice = pred_3d[:, :, z_slice]

        # Create subplots
        ncols = 3 if sigma_3d is not None else 2
        fig_size = figsize or (15, 5)
        fig, axes = plt.subplots(1, ncols, figsize=fig_size)

        # True data
        im1 = axes[0].imshow(true_slice.T, origin="lower", cmap="viridis")
        axes[0].set_title("True Data")
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Predicted data
        im2 = axes[1].imshow(pred_slice.T, origin="lower", cmap="viridis")
        axes[1].set_title("Predicted Data")
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Uncertainty if provided
        if sigma_3d is not None:
            sigma_slice = sigma_3d[:, :, z_slice]
            im3 = axes[2].imshow(sigma_slice.T, origin="lower", cmap="plasma")
            axes[2].set_title("Uncertainty")
            plt.colorbar(im3, ax=axes[2], shrink=0.8)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        return fig, axes

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        filename: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot residuals analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            filename: Optional filename to save plot
            figsize: Figure size override

        Returns:
            Tuple of (figure, axes array)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        residuals = y_true - y_pred

        fig_size = figsize or (12, 4)
        fig, axes = plt.subplots(1, 3, figsize=fig_size)

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        axes[0].set_xlabel("True Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].set_title("True vs Predicted")

        # Residuals vs predicted
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color="r", linestyle="--")
        axes[1].set_xlabel("Predicted Values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residuals vs Predicted")

        # Residuals histogram
        axes[2].hist(residuals, bins=30, alpha=0.7)
        axes[2].set_xlabel("Residuals")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Residuals Distribution")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        return fig, axes

    def plot_training_curve(
        self,
        iterations: np.ndarray,
        loss_values: np.ndarray,
        *,
        filename: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot training curve.

        Args:
            iterations: Iteration numbers
            loss_values: Loss values
            filename: Optional filename to save plot
            figsize: Figure size override

        Returns:
            Tuple of (figure, axes)
        """
        if len(iterations) != len(loss_values):
            raise ValueError("iterations and loss_values must have the same length")

        fig_size = figsize or (8, 6)
        fig, ax = plt.subplots(figsize=fig_size)

        ax.plot(iterations, loss_values, "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curve")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        return fig, ax

    def plot_feature_importance(
        self,
        feature_names: list[str],
        importance_values: np.ndarray,
        *,
        filename: str | Path | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot feature importance.

        Args:
            feature_names: List of feature names
            importance_values: Importance values
            filename: Optional filename to save plot
            figsize: Figure size override

        Returns:
            Tuple of (figure, axes)
        """
        if len(feature_names) != len(importance_values):
            raise ValueError(
                "feature_names and importance_values must have the same length"
            )

        fig_size = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=fig_size)

        # Sort by importance
        sorted_idx = np.argsort(importance_values)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = importance_values[sorted_idx]

        ax.barh(range(len(sorted_names)), sorted_values)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved: {filename}")

        return fig, ax

    def plot_comparison(
        self,
        original: np.ndarray,
        predicted: np.ndarray,
        uncertainty: np.ndarray | None = None,
        *,
        slice_idx: int = 0,
        titles: list[str] | None = None,
        filename: str | Path | None = None,
    ) -> None:
        """Plot comparison between original and predicted data.

        Args:
            original: Original data slice
            predicted: Predicted data slice
            uncertainty: Optional uncertainty data slice
            slice_idx: Slice index for title
            titles: Custom titles for subplots
            filename: Optional filename to save plot
        """
        n_plots = 3 if uncertainty is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

        if n_plots == 2:
            axes = [axes[0], axes[1]]

        # Default titles
        if titles is None:
            titles = [
                f"Original PERMX (Z={slice_idx})",
                f"Predicted PERMX (Z={slice_idx})",
                f"Uncertainty (Z={slice_idx})",
            ]

        # Original data
        im1 = axes[0].imshow(original.T, origin="lower", cmap="viridis")
        axes[0].set_title(titles[0], fontweight="bold")
        plt.colorbar(im1, ax=axes[0], label="mD", shrink=0.8)

        # Predicted data
        im2 = axes[1].imshow(predicted.T, origin="lower", cmap="viridis")
        axes[1].set_title(titles[1], fontweight="bold")
        plt.colorbar(im2, ax=axes[1], label="mD", shrink=0.8)

        # Uncertainty (if provided)
        if uncertainty is not None:
            im3 = axes[2].imshow(uncertainty.T, origin="lower", cmap="magma")
            axes[2].set_title(titles[2], fontweight="bold")
            plt.colorbar(im3, ax=axes[2], label="σ", shrink=0.8)

        for ax in axes:
            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Comparison plot saved: {filename}")

        plt.show()

    def plot_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        model_name: str = "Model",
        filename: str | Path | None = None,
    ) -> None:
        """Plot model performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            filename: Optional filename to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig)

        # Predictions vs actual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])
        ax1.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"{model_name}: Predicted vs Actual")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals vs predicted
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors[1])
        ax2.axhline(y=0, color="r", linestyle="--", lw=2)
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        ax2.grid(True, alpha=0.3)

        # Residual histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(residuals, bins=30, alpha=0.7, color=self.colors[2], edgecolor="black")
        ax3.set_xlabel("Residuals")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Residual Distribution")
        ax3.grid(True, alpha=0.3)

        # Q-Q plot
        ax4 = fig.add_subplot(gs[1, 1])
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title("Q-Q Plot (Normality Check)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Performance plot saved: {filename}")

        plt.show()

    def plot_training_history(
        self,
        losses: list[float],
        *,
        model_name: str = "Model",
        filename: str | Path | None = None,
    ) -> None:
        """Plot training loss history.

        Args:
            losses: List of loss values
            model_name: Name of the model
            filename: Optional filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = range(1, len(losses) + 1)
        ax.plot(
            iterations,
            losses,
            color=self.colors[0],
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} Training History")
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(losses) > 10:
            z = np.polyfit(iterations, losses, 1)
            p = np.poly1d(z)
            ax.plot(
                iterations,
                p(iterations),
                "--",
                color=self.colors[1],
                alpha=0.8,
                label="Trend",
            )
            ax.legend()

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Training history plot saved: {filename}")

        plt.show()

    def plot_model_comparison(
        self,
        results: dict[str, dict[str, float]],
        *,
        metrics: list[str] = ["r2", "rmse", "mae"],
        filename: str | Path | None = None,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
        """Plot comparison between multiple models.

        Args:
            results: Dictionary of model results
            metrics: List of metrics to plot
            filename: Optional filename to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        model_names = list(results.keys())

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]

            bars = axes[i].bar(
                model_names, values, color=self.colors[: len(model_names)]
            )
            axes[i].set_title(f"{metric.upper()} Comparison", fontweight="bold")
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Model comparison plot saved: {filename}")

        return fig, axes


# Convenience functions for quick plotting
def quick_slice_plot(
    data: np.ndarray, title: str, filename: str | None = None, **kwargs
) -> None:
    """Quick function to plot a 2D slice."""
    plotter = SPE9Plotter()
    plotter.plot_slice(data, title, filename=filename, **kwargs)


def quick_comparison_plot(
    original: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray | None = None,
    filename: str | None = None,
    **kwargs,
) -> None:
    """Quick function to plot comparison."""
    plotter = SPE9Plotter()
    plotter.plot_comparison(
        original, predicted, uncertainty, filename=filename, **kwargs
    )


if __name__ == "__main__":
    print("SPE9 Plotting Utilities")
    print("Clean, focused plotting module for geomodeling results")
    print("\nExample usage:")
    print("plotter = SPE9Plotter()")
    print("plotter.plot_slice(data, 'My Plot', filename='output.png')")
    print("plotter.plot_comparison(original, predicted, uncertainty)")
