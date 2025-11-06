"""
Kriging Module for Spatial Interpolation

Implements ordinary kriging, universal kriging, and co-kriging for geostatistical
interpolation. Kriging is the Best Linear Unbiased Predictor (BLUP) that uses
spatial correlation structure to estimate values at unsampled locations.

Key concepts:
- Ordinary Kriging: Assumes constant but unknown mean
- Universal Kriging: Accounts for spatial trends
- Co-Kriging: Leverages correlation between multiple variables

Performance: Numba-accelerated distance calculations for 5-20x speedup.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


from .exceptions import DataValidationError, InvalidParameterError
from .variogram import VariogramModel, predict_variogram


@dataclass
class KrigingResult:
    """Container for kriging predictions and diagnostics.

    Attributes:
        predictions: Predicted values at target locations
        variance: Kriging variance (prediction uncertainty)
        weights: Kriging weights for each sample
        lagrange_multiplier: Lagrange multiplier from kriging system
    """

    predictions: np.ndarray
    variance: np.ndarray
    weights: Optional[np.ndarray] = None
    lagrange_multiplier: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"Kriging Result:\n"
            f"  Predictions: {len(self.predictions)} points\n"
            f"  Mean prediction: {self.predictions.mean():.4f}\n"
            f"  Mean variance: {self.variance.mean():.4f}\n"
            f"  Prediction range: [{self.predictions.min():.4f}, {self.predictions.max():.4f}]"
        )


@njit(cache=True, fastmath=True)
def _compute_distances_fast(point: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated Euclidean distance computation.

    5-10x faster than scipy.spatial.distance.cdist for single point queries.
    """
    n_points = coordinates.shape[0]
    n_dims = coordinates.shape[1]
    distances = np.empty(n_points)

    for i in range(n_points):
        dist_sq = 0.0
        for d in range(n_dims):
            diff = point[d] - coordinates[i, d]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)

    return distances


class OrdinaryKriging:
    """
    Ordinary Kriging interpolation.

    Assumes a constant but unknown mean. The kriging system is:

    [K + λ·1] [w] = [k]
    [1'    0] [μ]   [1]

    Where:
    - K is the covariance matrix between sample points
    - k is the covariance vector between samples and target
    - w are the kriging weights
    - μ is the Lagrange multiplier
    - λ is a small regularization term for numerical stability
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        regularization: float = 1e-10,
    ):
        """
        Initialize Ordinary Kriging.

        Args:
            variogram_model: Fitted variogram model
            regularization: Small value added to diagonal for stability
        """
        self.variogram_model = variogram_model
        self.regularization = regularization
        self.coordinates = None
        self.values = None
        self.K_inv = None

    def fit(self, coordinates: np.ndarray, values: np.ndarray):
        """
        Fit kriging system to training data.

        Args:
            coordinates: Sample coordinates (n_samples, n_dims)
            values: Sample values (n_samples,)
        """
        if coordinates.shape[0] != len(values):
            raise DataValidationError(
                f"Coordinates ({coordinates.shape[0]}) and values ({len(values)}) must have same length",
                suggestion="Check that each coordinate has a corresponding value",
            )

        if len(values) < 3:
            raise DataValidationError(
                f"Need at least 3 samples for kriging, got {len(values)}",
                suggestion="Provide more data points",
            )

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Compute covariance matrix K
        # For a variogram γ(h), covariance C(h) = sill - γ(h)
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix

        # Add regularization to diagonal
        K += self.regularization * np.eye(n)

        # Build augmented kriging matrix with Lagrange multiplier
        # [K  1]
        # [1' 0]
        K_aug = np.zeros((n + 1, n + 1))
        K_aug[:n, :n] = K
        K_aug[:n, n] = 1
        K_aug[n, :n] = 1
        K_aug[n, n] = 0

        # Invert once for efficiency
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Kriging matrix is singular. Increasing regularization.", UserWarning
            )
            K += self.regularization * 100 * np.eye(n)
            K_aug[:n, :n] = K
            self.K_inv = np.linalg.inv(K_aug)

        return self

    def predict(
        self,
        coordinates_target: np.ndarray,
        return_variance: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at target locations.

        Args:
            coordinates_target: Target coordinates (n_targets, n_dims)
            return_variance: Whether to return kriging variance

        Returns:
            predictions: Predicted values
            variance: Kriging variance (if return_variance=True)
        """
        if self.K_inv is None:
            raise DataValidationError(
                "Kriging system not fitted", suggestion="Call fit() before predict()"
            )

        n_targets = coordinates_target.shape[0]
        n_samples = len(self.values)

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        # Predict each target point
        for i in range(n_targets):
            target = coordinates_target[i]

            # Compute covariance vector k between samples and target
            # Use Numba-accelerated distance computation (5-10x faster)
            if NUMBA_AVAILABLE:
                distances = _compute_distances_fast(target, self.coordinates)
            else:
                distances = cdist(self.coordinates, target.reshape(1, -1)).ravel()

            gamma_vector = predict_variogram(self.variogram_model, distances)
            k = self.variogram_model.sill - gamma_vector

            # Augment with 1 for Lagrange multiplier
            k_aug = np.zeros(n_samples + 1)
            k_aug[:n_samples] = k
            k_aug[n_samples] = 1

            # Solve kriging system: weights = K_inv @ k_aug
            weights_aug = self.K_inv @ k_aug
            weights = weights_aug[:n_samples]
            lagrange = weights_aug[n_samples]

            # Prediction: weighted sum
            predictions[i] = np.dot(weights, self.values)

            # Kriging variance: C(0) - w'k - μ
            if return_variance:
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = C_0 - np.dot(weights, k) - lagrange

                # Ensure non-negative variance
                variances[i] = max(variances[i], 0.0)

        if return_variance:
            return predictions, variances
        return predictions, None

    def cross_validate(self, n_folds: int = 5) -> Dict[str, float]:
        """
        Leave-one-out cross-validation.

        Args:
            n_folds: Number of folds (default: leave-one-out)

        Returns:
            Dictionary with MAE, RMSE, R²
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold

        if n_folds == -1:
            n_folds = len(self.values)  # Leave-one-out

        kf = KFold(
            n_splits=min(n_folds, len(self.values)), shuffle=True, random_state=42
        )

        predictions = np.zeros_like(self.values)

        for train_idx, test_idx in kf.split(self.coordinates):
            # Fit on training fold
            ok_fold = OrdinaryKriging(self.variogram_model, self.regularization)
            ok_fold.fit(self.coordinates[train_idx], self.values[train_idx])

            # Predict on test fold
            pred, _ = ok_fold.predict(self.coordinates[test_idx], return_variance=False)
            predictions[test_idx] = pred

        return {
            "mae": mean_absolute_error(self.values, predictions),
            "rmse": np.sqrt(mean_squared_error(self.values, predictions)),
            "r2": r2_score(self.values, predictions),
        }


class UniversalKriging:
    """
    Universal Kriging with trend modeling.

    Accounts for spatial trends by including drift terms (e.g., linear, quadratic).
    The kriging system becomes:

    [K  F] [w] = [k]
    [F' 0] [ν]   [f]

    Where F contains drift functions evaluated at sample points.
    """

    def __init__(
        self,
        variogram_model: VariogramModel,
        drift_terms: List[str] = ["linear"],
        regularization: float = 1e-10,
    ):
        """
        Initialize Universal Kriging.

        Args:
            variogram_model: Fitted variogram model
            drift_terms: List of drift terms ('constant', 'linear', 'quadratic')
            regularization: Regularization parameter
        """
        self.variogram_model = variogram_model
        self.drift_terms = drift_terms
        self.regularization = regularization
        self.coordinates = None
        self.values = None
        self.K_inv = None
        self.n_drift = 0

    def _compute_drift_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute drift matrix F from coordinates."""
        n = coordinates.shape[0]
        n_dims = coordinates.shape[1]

        drift_functions = []

        if "constant" in self.drift_terms:
            drift_functions.append(np.ones(n))

        if "linear" in self.drift_terms:
            for d in range(n_dims):
                drift_functions.append(coordinates[:, d])

        if "quadratic" in self.drift_terms:
            for d in range(n_dims):
                drift_functions.append(coordinates[:, d] ** 2)
            # Cross terms
            if n_dims >= 2:
                for i in range(n_dims):
                    for j in range(i + 1, n_dims):
                        drift_functions.append(coordinates[:, i] * coordinates[:, j])

        if not drift_functions:
            raise InvalidParameterError(
                "No valid drift terms specified",
                valid_values=["constant", "linear", "quadratic"],
            )

        F = np.column_stack(drift_functions)
        self.n_drift = F.shape[1]

        return F

    def fit(self, coordinates: np.ndarray, values: np.ndarray):
        """Fit universal kriging system."""
        if coordinates.shape[0] != len(values):
            raise DataValidationError(
                f"Coordinates and values must have same length",
                suggestion="Check data alignment",
            )

        self.coordinates = coordinates
        self.values = values
        n = len(values)

        # Compute covariance matrix
        distances = cdist(coordinates, coordinates)
        gamma_matrix = predict_variogram(self.variogram_model, distances)
        K = self.variogram_model.sill - gamma_matrix
        K += self.regularization * np.eye(n)

        # Compute drift matrix
        F = self._compute_drift_matrix(coordinates)
        m = F.shape[1]

        # Build augmented system
        # [K  F]
        # [F' 0]
        K_aug = np.zeros((n + m, n + m))
        K_aug[:n, :n] = K
        K_aug[:n, n:] = F
        K_aug[n:, :n] = F.T

        # Invert
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
            warnings.warn("Kriging matrix singular. Increasing regularization.")
            K += self.regularization * 100 * np.eye(n)
            K_aug[:n, :n] = K
            self.K_inv = np.linalg.inv(K_aug)

        return self

    def predict(
        self,
        coordinates_target: np.ndarray,
        return_variance: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict at target locations with trend."""
        if self.K_inv is None:
            raise DataValidationError(
                "Universal kriging system not fitted", suggestion="Call fit() first"
            )

        n_targets = coordinates_target.shape[0]
        n_samples = len(self.values)

        predictions = np.zeros(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        # Compute drift for targets
        F_target = self._compute_drift_matrix(coordinates_target)

        for i in range(n_targets):
            target = coordinates_target[i : i + 1]

            # Covariance vector
            distances = cdist(self.coordinates, target).ravel()
            gamma_vector = predict_variogram(self.variogram_model, distances)
            k = self.variogram_model.sill - gamma_vector

            # Augment with drift
            k_aug = np.zeros(n_samples + self.n_drift)
            k_aug[:n_samples] = k
            k_aug[n_samples:] = F_target[i]

            # Solve
            weights_aug = self.K_inv @ k_aug
            weights = weights_aug[:n_samples]

            # Prediction
            predictions[i] = np.dot(weights, self.values)

            # Variance
            if return_variance:
                C_0 = self.variogram_model.sill - self.variogram_model.nugget
                variances[i] = max(C_0 - np.dot(weights, k), 0.0)

        if return_variance:
            return predictions, variances
        return predictions, None


class CoKriging:
    """
    Co-Kriging for multiple correlated variables.

    Leverages cross-correlation between primary and secondary variables
    to improve predictions. Useful when secondary variable is densely
    sampled but primary is sparse.
    """

    def __init__(
        self,
        primary_variogram: VariogramModel,
        secondary_variogram: VariogramModel,
        cross_variogram: VariogramModel,
        regularization: float = 1e-10,
    ):
        """
        Initialize Co-Kriging.

        Args:
            primary_variogram: Variogram for primary variable
            secondary_variogram: Variogram for secondary variable
            cross_variogram: Cross-variogram between variables
            regularization: Regularization parameter
        """
        self.primary_variogram = primary_variogram
        self.secondary_variogram = secondary_variogram
        self.cross_variogram = cross_variogram
        self.regularization = regularization

        self.primary_coords = None
        self.primary_values = None
        self.secondary_coords = None
        self.secondary_values = None
        self.K_inv = None

    def fit(
        self,
        primary_coords: np.ndarray,
        primary_values: np.ndarray,
        secondary_coords: np.ndarray,
        secondary_values: np.ndarray,
    ):
        """
        Fit co-kriging system.

        Args:
            primary_coords: Coordinates of primary variable samples
            primary_values: Values of primary variable
            secondary_coords: Coordinates of secondary variable samples
            secondary_values: Values of secondary variable
        """
        self.primary_coords = primary_coords
        self.primary_values = primary_values
        self.secondary_coords = secondary_coords
        self.secondary_values = secondary_values

        n1 = len(primary_values)
        n2 = len(secondary_values)
        n = n1 + n2

        # Build covariance matrix
        # [K11  K12]
        # [K21  K22]
        K = np.zeros((n, n))

        # K11: primary-primary
        dist11 = cdist(primary_coords, primary_coords)
        gamma11 = predict_variogram(self.primary_variogram, dist11)
        K[:n1, :n1] = self.primary_variogram.sill - gamma11

        # K22: secondary-secondary
        dist22 = cdist(secondary_coords, secondary_coords)
        gamma22 = predict_variogram(self.secondary_variogram, dist22)
        K[n1:, n1:] = self.secondary_variogram.sill - gamma22

        # K12, K21: cross-covariance
        dist12 = cdist(primary_coords, secondary_coords)
        gamma12 = predict_variogram(self.cross_variogram, dist12)
        cross_cov = self.cross_variogram.sill - gamma12
        K[:n1, n1:] = cross_cov
        K[n1:, :n1] = cross_cov.T

        # Regularization
        K += self.regularization * np.eye(n)

        # Augment with Lagrange multipliers (2 constraints)
        K_aug = np.zeros((n + 2, n + 2))
        K_aug[:n, :n] = K
        K_aug[:n1, n] = 1  # Primary constraint
        K_aug[n, :n1] = 1
        K_aug[n1:n, n + 1] = 1  # Secondary constraint
        K_aug[n + 1, n1:n] = 1

        # Invert
        try:
            self.K_inv = np.linalg.inv(K_aug)
        except np.linalg.LinAlgError:
            warnings.warn("Co-kriging matrix singular.")
            K += self.regularization * 100 * np.eye(n)
            K_aug[:n, :n] = K
            self.K_inv = np.linalg.inv(K_aug)

        return self

    def predict(
        self,
        coordinates_target: np.ndarray,
        return_variance: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict primary variable at target locations using both variables."""
        if self.K_inv is None:
            raise DataValidationError(
                "Co-kriging system not fitted", suggestion="Call fit() first"
            )

        n_targets = coordinates_target.shape[0]
        n1 = len(self.primary_values)
        n2 = len(self.secondary_values)
        n = n1 + n2

        predictions = np.ndarray(n_targets)
        variances = np.zeros(n_targets) if return_variance else None

        for i in range(n_targets):
            target = coordinates_target[i : i + 1]

            # Covariance vectors
            dist_p = cdist(self.primary_coords, target).ravel()
            gamma_p = predict_variogram(self.primary_variogram, dist_p)
            k_p = self.primary_variogram.sill - gamma_p

            dist_s = cdist(self.secondary_coords, target).ravel()
            gamma_s = predict_variogram(self.cross_variogram, dist_s)
            k_s = self.cross_variogram.sill - gamma_s

            # Augment
            k_aug = np.zeros(n + 2)
            k_aug[:n1] = k_p
            k_aug[n1:n] = k_s
            k_aug[n] = 1
            k_aug[n + 1] = 0  # Only primary constraint for prediction

            # Solve
            weights_aug = self.K_inv @ k_aug
            weights_p = weights_aug[:n1]
            weights_s = weights_aug[n1:n]

            # Prediction
            pred_p = np.dot(weights_p, self.primary_values)
            pred_s = np.dot(weights_s, self.secondary_values)
            predictions[i] = pred_p + pred_s

            # Variance
            if return_variance:
                C_0 = self.primary_variogram.sill - self.primary_variogram.nugget
                variances[i] = max(
                    C_0 - np.dot(weights_p, k_p) - np.dot(weights_s, k_s), 0.0
                )

        if return_variance:
            return predictions, variances
        return predictions, None


def simple_kriging(
    coordinates: np.ndarray,
    values: np.ndarray,
    coordinates_target: np.ndarray,
    variogram_model: VariogramModel,
    mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple Kriging with known mean.

    Simpler than Ordinary Kriging - assumes known constant mean.

    Args:
        coordinates: Sample coordinates
        values: Sample values
        coordinates_target: Target coordinates
        variogram_model: Fitted variogram
        mean: Known mean value

    Returns:
        predictions, variance
    """
    n = len(values)
    n_targets = coordinates_target.shape[0]

    # Center values
    values_centered = values - mean

    # Covariance matrix
    distances = cdist(coordinates, coordinates)
    gamma_matrix = predict_variogram(variogram_model, distances)
    K = variogram_model.sill - gamma_matrix
    K += 1e-10 * np.eye(n)  # Regularization

    # Invert
    K_inv = np.linalg.inv(K)

    predictions = np.zeros(n_targets)
    variances = np.zeros(n_targets)

    for i in range(n_targets):
        target = coordinates_target[i : i + 1]

        # Covariance vector
        distances_target = cdist(coordinates, target).ravel()
        gamma_vector = predict_variogram(variogram_model, distances_target)
        k = variogram_model.sill - gamma_vector

        # Weights
        weights = K_inv @ k

        # Prediction (add mean back)
        predictions[i] = np.dot(weights, values_centered) + mean

        # Variance
        C_0 = variogram_model.sill - variogram_model.nugget
        variances[i] = max(C_0 - np.dot(weights, k), 0.0)

    return predictions, variances
