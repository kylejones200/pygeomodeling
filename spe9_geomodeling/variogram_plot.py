"""
Variogram Visualization Utilities

Use plots to check the fit. Clear models produce better spatial estimates.
"""

from typing import Optional

import logging
import matplotlib.pyplot as plt
import numpy as np
import signalplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .variogram import VariogramModel, predict_variogram

# Configure logging
logger = logging.getLogger(__name__)

# Apply style globally
signalplot.apply()


def plot_variogram(
    lags: np.ndarray,
    semi_variance: np.ndarray,
    model: Optional[VariogramModel] = None,
    n_pairs: Optional[np.ndarray] = None,
    title: str = "Semi-Variogram",
    ax: Optional[Axes] = None,
    show_model_params: bool = True,
) -> tuple[Figure, Axes]:
    """Plot experimental variogram with optional fitted model.

    Args:
        lags: Lag distances
        semi_variance: Semi-variance values
        model: Optional fitted model to overlay
        n_pairs: Optional number of pairs per lag (for point sizing)
        title: Plot title
        ax: Optional matplotlib axes
        show_model_params: Whether to show model parameters on plot

    Returns:
        Tuple of (figure, axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot experimental variogram
    if n_pairs is not None:
        # Size points by number of pairs
        sizes = 50 + 200 * (n_pairs / n_pairs.max())
        scatter = ax.scatter(
            lags,
            semi_variance,
            s=sizes,
            alpha=0.6,
            c="#555555",
            edgecolors="none",
            linewidth=0,
            label="Experimental",
            zorder=3,
        )
    else:
        ax.scatter(
            lags,
            semi_variance,
            s=100,
            alpha=0.6,
            c="#555555",
            edgecolors="none",
            linewidth=0,
            label="Experimental",
            zorder=3,
        )

    # Plot fitted model
    if model is not None:
        h_fine = np.linspace(0, lags.max() * 1.1, 200)
        gamma_model = predict_variogram(model, h_fine)

        ax.plot(
            h_fine,
            gamma_model,
            color=signalplot.ACCENT,
            linewidth=2,
            label=f"{model.model_type.capitalize()} Model",
            zorder=2,
        )

        # Mark key parameters
        ax.axhline(y=model.nugget, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=model.sill, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(
            x=model.range_param, color="gray", linestyle="--", alpha=0.5, linewidth=1
        )

        # Annotate parameters
        if show_model_params:
            y_max = semi_variance.max()
            ax.text(
                0.02 * lags.max(),
                model.nugget + 0.02 * y_max,
                f"Nugget = {model.nugget:.3f}",
                fontsize=9,
                color="gray",
            )
            ax.text(
                0.02 * lags.max(),
                model.sill - 0.05 * y_max,
                f"Sill = {model.sill:.3f}",
                fontsize=9,
                color="gray",
            )
            ax.text(
                model.range_param + 0.02 * lags.max(),
                0.02 * y_max,
                f"Range = {model.range_param:.3f}",
                fontsize=9,
                color="gray",
                rotation=90,
            )

            # Add R² to legend
            ax.text(
                0.98,
                0.98,
                f"R² = {model.r_squared:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    ax.set_xlabel("Distance (lag)")
    ax.set_ylabel("Semi-variance")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax


def plot_variogram_comparison(
    lags: np.ndarray,
    semi_variance: np.ndarray,
    models: list[VariogramModel],
    n_pairs: Optional[np.ndarray] = None,
    title: str = "Variogram Model Comparison",
) -> tuple[Figure, Axes]:
    """Compare multiple variogram models.

    Args:
        lags: Lag distances
        semi_variance: Semi-variance values
        models: List of fitted models to compare
        n_pairs: Optional number of pairs per lag
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot experimental variogram
    if n_pairs is not None:
        sizes = 50 + 200 * (n_pairs / n_pairs.max())
        ax.scatter(
            lags,
            semi_variance,
            s=sizes,
            alpha=0.6,
            c="steelblue",
            edgecolors="black",
            linewidth=1,
            label="Experimental",
            zorder=3,
        )
    else:
        ax.scatter(
            lags,
            semi_variance,
            s=100,
            alpha=0.6,
            c="steelblue",
            edgecolors="black",
            linewidth=1,
            label="Experimental",
            zorder=3,
        )

    # Plot each model
    h_fine = np.linspace(0, lags.max() * 1.1, 200)

    # Use signalplot's restrained color logic
    for i, model in enumerate(models):
        gamma_model = predict_variogram(model, h_fine)
        # First model gets accent color, others get grays
        if i == 0:
            color = signalplot.ACCENT
            linewidth = 2.5
        else:
            gray_val = min(0.3 + (i * 0.15), 0.7)
            color = f"{gray_val}"
            linewidth = 1.5
            
        label = f"{model.model_type.capitalize()} (R²={model.r_squared:.3f})"
        ax.plot(h_fine, gamma_model, color=color, linewidth=linewidth, label=label, zorder=2)

    ax.set_xlabel("Distance (lag)")
    ax.set_ylabel("Semi-variance")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax


def plot_directional_variograms(
    coordinates: np.ndarray,
    values: np.ndarray,
    directions: list[float] = [0, 45, 90, 135],
    tolerance: float = 22.5,
    n_lags: int = 15,
    title: str = "Directional Variograms (Anisotropy Check)",
) -> tuple[Figure, Axes]:
    """Plot variograms in multiple directions to check for anisotropy.

    Args:
        coordinates: Spatial coordinates (n_samples, 2)
        values: Property values
        directions: List of directions in degrees (0 = East, 90 = North)
        tolerance: Angular tolerance
        n_lags: Number of lag bins
        title: Plot title

    Returns:
        Tuple of (figure, axes)
    """
    from .variogram import directional_variogram

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use a gray scale with accent for directions
    for i, direction in enumerate(directions):
        try:
            lags, sv, n_pairs = directional_variogram(
                coordinates, values, direction, tolerance, n_lags
            )

            if len(lags) > 0:
                if i == 0:
                    color = signalplot.ACCENT
                else:
                    gray_val = min(0.3 + (i * 0.15), 0.7)
                    color = f"{gray_val}"
                    
                ax.plot(
                    lags,
                    sv,
                    "o-",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=f"{direction}° ± {tolerance}°",
                    alpha=0.8,
                )
        except Exception as e:
            logger.warning(
                "Could not compute variogram for direction %d: %s",
                direction, e
            )

    ax.set_xlabel("Distance (lag)")
    ax.set_ylabel("Semi-variance")
    ax.set_title(title)
    ax.legend(loc="lower right")

    # Add compass rose
    ax_inset = fig.add_axes([0.15, 0.7, 0.15, 0.15])
    ax_inset.set_xlim(-1.2, 1.2)
    ax_inset.set_ylim(-1.2, 1.2)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")

    # Draw compass
    for i, direction in enumerate(directions):
        angle_rad = np.radians(90 - direction)  # Convert to math convention
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        
        if i == 0:
            color = signalplot.ACCENT
        else:
            gray_val = min(0.3 + (i * 0.15), 0.7)
            color = f"{gray_val}"
            
        ax_inset.arrow(
            0,
            0,
            x,
            y,
            head_width=0.15,
            head_length=0.15,
            fc=color,
            ec=color,
            linewidth=2,
            alpha=0.7,
        )

    # Add N/E/S/W labels
    ax_inset.text(0, 1.1, "N", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_inset.text(1.1, 0, "E", ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig, ax


def plot_variogram_cloud(
    coordinates: np.ndarray,
    values: np.ndarray,
    max_pairs: int = 5000,
    title: str = "Variogram Cloud",
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot variogram cloud (all pairwise semi-variances).

    Useful for identifying outliers before fitting.

    Args:
        coordinates: Spatial coordinates
        values: Property values
        max_pairs: Maximum number of pairs to plot (for performance)
        title: Plot title
        ax: Optional matplotlib axes

    Returns:
        Tuple of (figure, axes)
    """
    from scipy.spatial.distance import pdist

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Compute all pairwise distances and semi-variances
    distances = pdist(coordinates)
    value_diffs = pdist(values.reshape(-1, 1))
    semi_variances = 0.5 * value_diffs**2

    # Subsample if too many pairs
    if len(distances) > max_pairs:
        indices = np.random.choice(len(distances), max_pairs, replace=False)
        distances = distances[indices]
        semi_variances = semi_variances[indices]

    # Plot cloud
    ax.scatter(distances, semi_variances, alpha=0.3, s=10, c="#555555")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Semi-variance")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax
