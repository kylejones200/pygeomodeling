"""
Variogram Analysis Module

Semi-variograms help us see how similarity changes with distance. They are a core
tool for spatial analysis. The curve shows semi-variance on the vertical axis and
distance on the horizontal axis. The shape tells us where spatial influence fades.

Key concepts:
- Nugget: Small-scale noise or measurement error
- Sill: Plateau where distance no longer adds variance
- Range: Distance where correlation becomes weak
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
import warnings

from .exceptions import DataValidationError, InvalidParameterError


@dataclass
class VariogramModel:
    """Container for fitted variogram model parameters.
    
    Attributes:
        model_type: Type of model ('spherical', 'exponential', 'gaussian')
        nugget: Nugget effect (small-scale variance)
        sill: Total sill (nugget + partial sill)
        range_param: Range parameter (correlation length)
        partial_sill: Partial sill (sill - nugget)
        r_squared: Goodness of fit
    """
    model_type: str
    nugget: float
    sill: float
    range_param: float
    partial_sill: float
    r_squared: float
    
    def __str__(self) -> str:
        return (
            f"{self.model_type.capitalize()} Variogram Model:\n"
            f"  Nugget: {self.nugget:.4f}\n"
            f"  Sill: {self.sill:.4f}\n"
            f"  Range: {self.range_param:.4f}\n"
            f"  R²: {self.r_squared:.4f}"
        )


def _spherical_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Spherical variogram model.
    
    Rises and then levels off near the range. Good for many natural processes.
    
    Args:
        h: Distance (lag)
        nugget: Nugget effect
        sill: Total sill
        range_param: Range parameter
        
    Returns:
        Semi-variance values
    """
    gamma = np.zeros_like(h, dtype=float)
    
    # For h < range: nugget + (sill - nugget) * [1.5*(h/a) - 0.5*(h/a)^3]
    mask = h < range_param
    if np.any(mask):
        h_scaled = h[mask] / range_param
        gamma[mask] = nugget + (sill - nugget) * (1.5 * h_scaled - 0.5 * h_scaled**3)
    
    # For h >= range: sill
    gamma[~mask] = sill
    
    return gamma


def _exponential_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Exponential variogram model.
    
    Rises fast and then approaches a limit. Never quite reaches the sill.
    
    Args:
        h: Distance (lag)
        nugget: Nugget effect
        sill: Total sill
        range_param: Range parameter (practical range ~3*range_param)
        
    Returns:
        Semi-variance values
    """
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))


def _gaussian_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """Gaussian variogram model.
    
    Rises slowly at first and then speeds up. Very smooth at the origin.
    
    Args:
        h: Distance (lag)
        nugget: Nugget effect
        sill: Total sill
        range_param: Range parameter
        
    Returns:
        Semi-variance values
    """
    return nugget + (sill - nugget) * (1 - np.exp(-(h**2) / (range_param**2)))


def _linear_model(h: np.ndarray, nugget: float, slope: float) -> np.ndarray:
    """Linear variogram model (no sill).
    
    Continues to increase linearly. Used when no clear range exists.
    
    Args:
        h: Distance (lag)
        nugget: Nugget effect
        slope: Slope of the line
        
    Returns:
        Semi-variance values
    """
    return nugget + slope * h


# Model registry
VARIOGRAM_MODELS: Dict[str, Callable] = {
    'spherical': _spherical_model,
    'exponential': _exponential_model,
    'gaussian': _gaussian_model,
    'linear': _linear_model,
}


def compute_experimental_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    lag_tolerance: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental semi-variogram from sample pairs.
    
    Bins the distances and computes average semi-variance by bin.
    
    Args:
        coordinates: Array of shape (n_samples, n_dims) with spatial coordinates
        values: Array of shape (n_samples,) with property values
        n_lags: Number of lag bins
        max_lag: Maximum lag distance (default: half of max distance)
        lag_tolerance: Tolerance for binning (fraction of lag width)
        
    Returns:
        Tuple of (lags, semi_variance, n_pairs) for each bin
        
    Raises:
        DataValidationError: If inputs are invalid
    """
    # Validate inputs
    if coordinates.ndim != 2:
        raise DataValidationError(
            f"Coordinates must be 2D array, got shape {coordinates.shape}",
            suggestion="Reshape coordinates to (n_samples, n_dimensions)"
        )
    
    if values.ndim != 1:
        raise DataValidationError(
            f"Values must be 1D array, got shape {values.shape}",
            suggestion="Flatten values array"
        )
    
    if len(coordinates) != len(values):
        raise DataValidationError(
            f"Coordinates ({len(coordinates)}) and values ({len(values)}) must have same length",
            suggestion="Check that each coordinate has a corresponding value"
        )
    
    if len(values) < 10:
        raise DataValidationError(
            f"Need at least 10 samples for variogram, got {len(values)}",
            suggestion="Provide more data points"
        )
    
    # Compute pairwise distances
    distances = pdist(coordinates)
    
    # Compute pairwise squared differences
    value_diffs = pdist(values.reshape(-1, 1))
    semi_variance_pairs = 0.5 * value_diffs**2
    
    # Determine lag bins
    if max_lag is None:
        max_lag = distances.max() / 2.0
    
    lag_width = max_lag / n_lags
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    
    # Bin the pairs and compute average semi-variance
    lags = []
    semi_variances = []
    n_pairs_list = []
    
    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]
        
        # Find pairs in this lag bin (with tolerance)
        tolerance = lag_tolerance * lag_width
        mask = (distances >= lag_min - tolerance) & (distances < lag_max + tolerance)
        
        if np.sum(mask) > 0:
            lags.append(lag_centers[i])
            semi_variances.append(np.mean(semi_variance_pairs[mask]))
            n_pairs_list.append(np.sum(mask))
    
    return np.array(lags), np.array(semi_variances), np.array(n_pairs_list)


def fit_variogram_model(
    lags: np.ndarray,
    semi_variance: np.ndarray,
    model_type: str = 'spherical',
    nugget_init: Optional[float] = None,
    sill_init: Optional[float] = None,
    range_init: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> VariogramModel:
    """Fit a variogram model to experimental data.
    
    Choose a model that fits the process. Keep the method simple at first.
    Use a plot to check the fit. Avoid overfitting to noise.
    
    Args:
        lags: Lag distances
        semi_variance: Semi-variance values
        model_type: Type of model ('spherical', 'exponential', 'gaussian', 'linear')
        nugget_init: Initial nugget value (default: min semi-variance)
        sill_init: Initial sill value (default: max semi-variance)
        range_init: Initial range value (default: 2/3 of max lag)
        weights: Optional weights for fitting (e.g., sqrt of n_pairs)
        
    Returns:
        Fitted VariogramModel
        
    Raises:
        InvalidParameterError: If model_type is invalid
        DataValidationError: If fitting fails
    """
    if model_type not in VARIOGRAM_MODELS:
        raise InvalidParameterError(
            f"Unknown model type: {model_type}",
            valid_values=list(VARIOGRAM_MODELS.keys())
        )
    
    if len(lags) < 3:
        raise DataValidationError(
            f"Need at least 3 lag bins for fitting, got {len(lags)}",
            suggestion="Use more lag bins or provide more data"
        )
    
    # Set initial parameter guesses
    if nugget_init is None:
        nugget_init = semi_variance[0] if semi_variance[0] > 0 else 0.01
    
    if sill_init is None:
        sill_init = np.max(semi_variance)
    
    if range_init is None:
        range_init = lags[-1] * 0.67  # 2/3 of max lag
    
    # Get the model function
    model_func = VARIOGRAM_MODELS[model_type]
    
    # Fit the model
    try:
        if model_type == 'linear':
            # Linear model: only nugget and slope
            p0 = [nugget_init, (sill_init - nugget_init) / range_init]
            bounds = ([0, 0], [np.inf, np.inf])
            
            popt, _ = curve_fit(
                model_func,
                lags,
                semi_variance,
                p0=p0,
                bounds=bounds,
                sigma=1.0 / weights if weights is not None else None,
                maxfev=10000
            )
            
            nugget, slope = popt
            sill = semi_variance[-1]  # Use last value as approximate sill
            range_param = range_init
            partial_sill = sill - nugget
            
        else:
            # Bounded models: nugget, sill, range
            p0 = [nugget_init, sill_init, range_init]
            bounds = ([0, nugget_init, 0], [sill_init, np.inf, lags[-1] * 2])
            
            popt, _ = curve_fit(
                model_func,
                lags,
                semi_variance,
                p0=p0,
                bounds=bounds,
                sigma=1.0 / weights if weights is not None else None,
                maxfev=10000
            )
            
            nugget, sill, range_param = popt
            partial_sill = sill - nugget
        
        # Compute R²
        predictions = model_func(lags, *popt)
        ss_res = np.sum((semi_variance - predictions)**2)
        ss_tot = np.sum((semi_variance - np.mean(semi_variance))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return VariogramModel(
            model_type=model_type,
            nugget=nugget,
            sill=sill,
            range_param=range_param,
            partial_sill=partial_sill,
            r_squared=r_squared
        )
        
    except Exception as e:
        raise DataValidationError(
            f"Failed to fit {model_type} variogram model: {str(e)}",
            suggestion="Try different initial parameters or a different model type"
        )


def predict_variogram(
    model: VariogramModel,
    distances: np.ndarray
) -> np.ndarray:
    """Predict semi-variance at given distances using fitted model.
    
    Args:
        model: Fitted VariogramModel
        distances: Distances at which to predict
        
    Returns:
        Predicted semi-variance values
    """
    model_func = VARIOGRAM_MODELS[model.model_type]
    
    if model.model_type == 'linear':
        slope = model.partial_sill / model.range_param
        return model_func(distances, model.nugget, slope)
    else:
        return model_func(distances, model.nugget, model.sill, model.range_param)


def directional_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    direction: float = 0.0,
    tolerance: float = 22.5,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute directional (anisotropic) variogram.
    
    Useful for detecting anisotropy in spatial correlation.
    
    Args:
        coordinates: Array of shape (n_samples, 2) with x, y coordinates
        values: Array of shape (n_samples,) with property values
        direction: Direction in degrees (0 = East, 90 = North)
        tolerance: Angular tolerance in degrees
        n_lags: Number of lag bins
        max_lag: Maximum lag distance
        
    Returns:
        Tuple of (lags, semi_variance, n_pairs) for specified direction
        
    Raises:
        DataValidationError: If coordinates are not 2D
    """
    if coordinates.shape[1] != 2:
        raise DataValidationError(
            f"Directional variogram requires 2D coordinates, got {coordinates.shape[1]}D",
            suggestion="Use only x and y coordinates"
        )
    
    # Compute all pairwise vectors
    n = len(coordinates)
    indices = np.triu_indices(n, k=1)
    
    vectors = coordinates[indices[1]] - coordinates[indices[0]]
    distances = np.linalg.norm(vectors, axis=1)
    
    # Compute angles (in degrees)
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = (angles + 360) % 360  # Normalize to [0, 360)
    
    # Filter by direction
    direction = direction % 360
    angle_diff = np.abs(angles - direction)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # Handle wrap-around
    
    mask = angle_diff <= tolerance
    
    if np.sum(mask) < 10:
        warnings.warn(
            f"Only {np.sum(mask)} pairs found in direction {direction}° ± {tolerance}°. "
            "Consider increasing tolerance or using omnidirectional variogram."
        )
    
    # Compute semi-variance for filtered pairs
    value_diffs = values[indices[1][mask]] - values[indices[0][mask]]
    semi_variance_pairs = 0.5 * value_diffs**2
    filtered_distances = distances[mask]
    
    # Bin and average
    if max_lag is None:
        max_lag = filtered_distances.max() / 2.0 if len(filtered_distances) > 0 else 1.0
    
    lag_width = max_lag / n_lags
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    
    lags = []
    semi_variances = []
    n_pairs_list = []
    
    for i in range(n_lags):
        lag_mask = (filtered_distances >= lag_bins[i]) & (filtered_distances < lag_bins[i + 1])
        
        if np.sum(lag_mask) > 0:
            lags.append(lag_centers[i])
            semi_variances.append(np.mean(semi_variance_pairs[lag_mask]))
            n_pairs_list.append(np.sum(lag_mask))
    
    return np.array(lags), np.array(semi_variances), np.array(n_pairs_list)


def cross_validation_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    model: VariogramModel,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Cross-validate variogram model.
    
    Simple check to confirm the model with cross-validation.
    
    Args:
        coordinates: Spatial coordinates
        values: Property values
        model: Fitted variogram model
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with validation metrics (mae, rmse, r2)
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_true = []
    all_pred = []
    
    for train_idx, test_idx in kf.split(coordinates):
        # Fit on training data
        lags_train, sv_train, _ = compute_experimental_variogram(
            coordinates[train_idx],
            values[train_idx]
        )
        
        # Predict on test data
        lags_test, sv_test, _ = compute_experimental_variogram(
            coordinates[test_idx],
            values[test_idx]
        )
        
        sv_pred = predict_variogram(model, lags_test)
        
        all_true.extend(sv_test)
        all_pred.extend(sv_pred)
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    return {
        'mae': mean_absolute_error(all_true, all_pred),
        'rmse': np.sqrt(mean_squared_error(all_true, all_pred)),
        'r2': r2_score(all_true, all_pred)
    }
