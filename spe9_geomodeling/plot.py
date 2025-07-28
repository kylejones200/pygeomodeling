"""Clean plotting utilities for SPE9 geomodeling.

Focused plotting module with proper separation of concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SPE9Plotter:
    """Clean plotting utilities for SPE9 geomodeling results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
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
        filename: Optional[Union[str, Path]] = None,
        cmap: str = "viridis",
        log_scale: bool = False,
        colorbar_label: str = "Value",
        figsize: Optional[Tuple[int, int]] = None
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
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Index')
        ax.set_ylabel('Y Index')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved: {filename}")
        
        plt.show()
    
    def plot_comparison(
        self,
        original: np.ndarray,
        predicted: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        *,
        slice_idx: int = 0,
        titles: Optional[List[str]] = None,
        filename: Optional[Union[str, Path]] = None
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
                f'Original PERMX (Z={slice_idx})',
                f'Predicted PERMX (Z={slice_idx})',
                f'Uncertainty (Z={slice_idx})'
            ]
        
        # Original data
        im1 = axes[0].imshow(original.T, origin='lower', cmap='viridis')
        axes[0].set_title(titles[0], fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='mD', shrink=0.8)
        
        # Predicted data
        im2 = axes[1].imshow(predicted.T, origin='lower', cmap='viridis')
        axes[1].set_title(titles[1], fontweight='bold')
        plt.colorbar(im2, ax=axes[1], label='mD', shrink=0.8)
        
        # Uncertainty (if provided)
        if uncertainty is not None:
            im3 = axes[2].imshow(uncertainty.T, origin='lower', cmap='magma')
            axes[2].set_title(titles[2], fontweight='bold')
            plt.colorbar(im3, ax=axes[2], label='σ', shrink=0.8)
        
        for ax in axes:
            ax.set_xlabel('X Index')
            ax.set_ylabel('Y Index')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Comparison plot saved: {filename}")
        
        plt.show()
    
    def plot_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        model_name: str = "Model",
        filename: Optional[Union[str, Path]] = None
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
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name}: Predicted vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals vs predicted
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors[1])
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Residual histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(residuals, bins=30, alpha=0.7, color=self.colors[2], edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax4 = fig.add_subplot(gs[1, 1])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality Check)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Performance plot saved: {filename}")
        
        plt.show()
    
    def plot_training_history(
        self,
        losses: List[float],
        *,
        model_name: str = "Model",
        filename: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot training loss history.
        
        Args:
            losses: List of loss values
            model_name: Name of the model
            filename: Optional filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(losses) + 1)
        ax.plot(iterations, losses, color=self.colors[0], linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} Training History')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(losses) > 10:
            z = np.polyfit(iterations, losses, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), "--", color=self.colors[1], alpha=0.8, label='Trend')
            ax.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Training history plot saved: {filename}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        *,
        metrics: List[str] = ['r2', 'rmse', 'mae'],
        filename: Optional[Union[str, Path]] = None
    ) -> None:
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
            
            bars = axes[i].bar(model_names, values, color=self.colors[:len(model_names)])
            axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Model comparison plot saved: {filename}")
        
        plt.show()


# Convenience functions for quick plotting
def quick_slice_plot(
    data: np.ndarray, 
    title: str, 
    filename: Optional[str] = None,
    **kwargs
) -> None:
    """Quick function to plot a 2D slice."""
    plotter = SPE9Plotter()
    plotter.plot_slice(data, title, filename=filename, **kwargs)


def quick_comparison_plot(
    original: np.ndarray,
    predicted: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    **kwargs
) -> None:
    """Quick function to plot comparison."""
    plotter = SPE9Plotter()
    plotter.plot_comparison(original, predicted, uncertainty, filename=filename, **kwargs)


if __name__ == "__main__":
    print("SPE9 Plotting Utilities")
    print("Clean, focused plotting module for geomodeling results")
    print("\nExample usage:")
    print("plotter = SPE9Plotter()")
    print("plotter.plot_slice(data, 'My Plot', filename='output.png')")
    print("plotter.plot_comparison(original, predicted, uncertainty)")
