"""
Formation Tops Detection Module

Automated detection and classification of formation boundaries from well logs.
Combines signal processing with machine learning to identify depth intervals
where geological units transition.

Implements boundary detection strategies described in automated well log
interpretation workflows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .exceptions import DataValidationError


@dataclass
class FormationTop:
    """Represents a formation boundary/top."""

    depth: float
    formation_name: str
    confidence: float  # 0-1
    method: str  # 'manual', 'signal_processing', 'ml'
    log_character: Dict[str, float]  # Average log values in formation

    def __str__(self) -> str:
        return (
            f"{self.formation_name} @ {self.depth:.1f}m "
            f"(confidence: {self.confidence:.2f}, method: {self.method})"
        )


@dataclass
class BoundaryDetectionResult:
    """Results from boundary detection analysis."""

    detected_boundaries: List[float]  # Depths of detected boundaries
    boundary_scores: np.ndarray  # Strength scores for each depth
    recommended_tops: List[FormationTop]
    processing_log: List[str]

    def __str__(self) -> str:
        return (
            f"Boundary Detection:\n"
            f"  Detected: {len(self.detected_boundaries)} boundaries\n"
            f"  Recommended tops: {len(self.recommended_tops)}\n"
            f"  Processing steps: {len(self.processing_log)}"
        )


class FormationTopDetector:
    """
    Automated formation top detection from well logs.

    Uses signal processing and machine learning to identify formation boundaries
    and classify them as true formation contacts vs. intra-formational variations.
    """

    def __init__(
        self,
        null_value: float = -999.25,
        min_formation_thickness: float = 5.0,  # meters
        boundary_threshold: float = 0.3,
    ):
        """
        Initialize formation top detector.

        Args:
            null_value: Value indicating missing data
            min_formation_thickness: Minimum thickness for a formation (meters)
            boundary_threshold: Threshold for boundary detection score (0-1)
        """
        self.null_value = null_value
        self.min_formation_thickness = min_formation_thickness
        self.boundary_threshold = boundary_threshold
        self.processing_log = []
        self.boundary_classifier = None
        self.scaler = StandardScaler()

    def compute_boundary_score(
        self,
        data: pd.DataFrame,
        curves: Optional[List[str]] = None,
        method: str = "composite",
    ) -> np.ndarray:
        """
        Compute boundary detection score for each depth.

        Higher scores indicate more likely formation boundaries.

        Args:
            data: DataFrame with log curves
            curves: List of curves to use (None = all)
            method: 'composite', 'gradient', or 'variance'

        Returns:
            Array of boundary scores (0-1) for each depth
        """
        if curves is None:
            curves = [
                col
                for col in data.columns
                if not col.endswith(("_DERIV", "_RATIO", "_ROLL", "_SPATIAL"))
            ]

        depth_step = np.median(np.diff(data.index))
        scores_by_curve = []

        for curve in curves:
            if curve not in data.columns:
                continue

            values = data[curve].values.copy()
            valid_mask = values != self.null_value

            if valid_mask.sum() < 10:
                continue

            # Replace nulls with interpolated values for processing
            values[~valid_mask] = np.nan
            values = pd.Series(values).interpolate(limit=5).values

            if method in ["composite", "gradient"]:
                # Gradient-based detection
                smoothed = gaussian_filter1d(values, sigma=2.0)
                gradient = np.abs(np.gradient(smoothed, depth_step))
                gradient_score = gradient / (gradient.max() + 1e-10)
                scores_by_curve.append(gradient_score)

            if method in ["composite", "variance"]:
                # Variance-based detection (local variability change)
                window = int(self.min_formation_thickness / depth_step)
                window = max(5, window)

                # Compute local variance
                local_var = (
                    pd.Series(values)
                    .rolling(window=window, center=True, min_periods=1)
                    .var()
                    .values
                )

                # Change in variance indicates boundary
                var_gradient = np.abs(np.gradient(local_var, depth_step))
                var_score = var_gradient / (var_gradient.max() + 1e-10)
                scores_by_curve.append(var_score)

        if not scores_by_curve:
            return np.zeros(len(data))

        # Combine scores across curves
        combined_scores = np.mean(scores_by_curve, axis=0)

        # Normalize to 0-1
        combined_scores = (combined_scores - combined_scores.min()) / (
            combined_scores.max() - combined_scores.min() + 1e-10
        )

        # Apply smoothing to reduce noise
        combined_scores = gaussian_filter1d(combined_scores, sigma=1.0)

        return combined_scores

    def detect_boundaries(
        self,
        data: pd.DataFrame,
        curves: Optional[List[str]] = None,
        use_peak_detection: bool = True,
    ) -> List[float]:
        """
        Detect formation boundaries from log data.

        Args:
            data: DataFrame with log curves
            curves: List of curves to use
            use_peak_detection: Use peak detection on boundary scores

        Returns:
            List of depths where boundaries detected
        """
        self.processing_log = ["Starting boundary detection"]

        # Compute boundary scores
        scores = self.compute_boundary_score(data, curves)

        if use_peak_detection:
            # Find peaks in boundary scores
            # Distance ensures minimum formation thickness
            min_distance = int(
                self.min_formation_thickness / np.median(np.diff(data.index))
            )

            peaks, properties = signal.find_peaks(
                scores,
                height=self.boundary_threshold,
                distance=min_distance,
                prominence=0.1,
            )

            boundary_depths = data.index[peaks].tolist()
            self.processing_log.append(
                f"Detected {len(peaks)} peaks above threshold {self.boundary_threshold}"
            )
        else:
            # Simple threshold crossing
            above_threshold = scores > self.boundary_threshold

            # Find transitions from below to above threshold
            transitions = np.diff(above_threshold.astype(int))
            boundary_indices = np.where(transitions == 1)[0] + 1

            boundary_depths = data.index[boundary_indices].tolist()
            self.processing_log.append(
                f"Detected {len(boundary_depths)} threshold crossings"
            )

        return boundary_depths

    def train_boundary_classifier(
        self,
        training_data: List[Tuple[pd.DataFrame, List[FormationTop]]],
        curves: Optional[List[str]] = None,
    ):
        """
        Train ML classifier to distinguish true boundaries from noise.

        Args:
            training_data: List of (log_data, formation_tops) tuples
            curves: List of curves to use as features
        """
        self.processing_log.append("Training boundary classifier")

        features_list = []
        labels_list = []

        for well_data, formation_tops in training_data:
            if curves is None:
                curves = [
                    col
                    for col in well_data.columns
                    if not col.endswith(("_DERIV", "_RATIO", "_ROLL", "_SPATIAL"))
                ]

            # Get boundary scores
            scores = self.compute_boundary_score(well_data, curves)

            # Create features at each depth
            for idx in range(len(well_data)):
                depth = well_data.index[idx]

                # Extract features: log values + boundary score + context
                feature_vector = []
                for curve in curves:
                    if curve in well_data.columns:
                        feature_vector.append(well_data[curve].iloc[idx])

                feature_vector.append(scores[idx])

                # Add context (average in window above/below)
                window_size = max(
                    1,
                    int(
                        self.min_formation_thickness
                        / 2
                        / np.median(np.diff(well_data.index))
                    ),
                )
                start_idx = max(0, idx - window_size)
                end_idx = min(len(well_data), idx + window_size + 1)

                for curve in curves:
                    if curve in well_data.columns:
                        above_avg = well_data[curve].iloc[start_idx:idx].mean()
                        below_avg = well_data[curve].iloc[idx + 1 : end_idx].mean()
                        feature_vector.extend([above_avg, below_avg])

                # Label: is this a formation boundary?
                is_boundary = any(
                    abs(depth - top.depth) < self.min_formation_thickness / 2
                    for top in formation_tops
                )

                features_list.append(feature_vector)
                labels_list.append(1 if is_boundary else 0)

        X = np.array(features_list)
        y = np.array(labels_list)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.boundary_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        )
        self.boundary_classifier.fit(X_scaled, y)

        self.processing_log.append(
            f"Trained classifier on {len(training_data)} wells, "
            f"{y.sum()} boundary samples, {len(y) - y.sum()} non-boundary samples"
        )

    def classify_boundaries(
        self,
        data: pd.DataFrame,
        detected_boundaries: List[float],
        curves: Optional[List[str]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Classify detected boundaries using trained ML model.

        Args:
            data: DataFrame with log curves
            detected_boundaries: List of candidate boundary depths
            curves: List of curves to use

        Returns:
            List of (depth, confidence) tuples for validated boundaries
        """
        if self.boundary_classifier is None:
            raise DataValidationError(
                "Boundary classifier not trained",
                "Call train_boundary_classifier() first",
            )

        if curves is None:
            curves = [
                col
                for col in data.columns
                if not col.endswith(("_DERIV", "_RATIO", "_ROLL", "_SPATIAL"))
            ]

        scores = self.compute_boundary_score(data, curves)
        validated_boundaries = []

        for depth in detected_boundaries:
            # Find nearest depth in data
            idx = (data.index - depth).abs().argmin()
            actual_depth = data.index[idx]

            # Extract features
            feature_vector = []
            for curve in curves:
                if curve in data.columns:
                    feature_vector.append(data[curve].iloc[idx])

            feature_vector.append(scores[idx])

            # Add context
            window_size = max(
                1,
                int(self.min_formation_thickness / 2 / np.median(np.diff(data.index))),
            )
            start_idx = max(0, idx - window_size)
            end_idx = min(len(data), idx + window_size + 1)

            for curve in curves:
                if curve in data.columns:
                    above_avg = data[curve].iloc[start_idx:idx].mean()
                    below_avg = data[curve].iloc[idx + 1 : end_idx].mean()
                    feature_vector.extend([above_avg, below_avg])

            # Predict
            X = np.array(feature_vector).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = self.scaler.transform(X)

            confidence = self.boundary_classifier.predict_proba(X_scaled)[0, 1]

            if confidence > 0.5:
                validated_boundaries.append((actual_depth, confidence))

        self.processing_log.append(
            f"Validated {len(validated_boundaries)}/{len(detected_boundaries)} boundaries"
        )

        return validated_boundaries

    def correlate_with_stratigraphy(
        self,
        boundaries: List[float],
        reference_sequence: List[str],
        regional_tops: Optional[Dict[str, float]] = None,
    ) -> List[FormationTop]:
        """
        Correlate detected boundaries with known stratigraphic sequence.

        Args:
            boundaries: List of boundary depths
            reference_sequence: Expected formation sequence (top to bottom)
            regional_tops: Optional dict of {formation_name: expected_depth}

        Returns:
            List of FormationTop objects with formation names
        """
        formation_tops = []

        if regional_tops:
            # Use regional knowledge to guide correlation
            for i, boundary_depth in enumerate(boundaries):
                # Find closest formation in regional tops
                best_match = None
                min_distance = float("inf")

                for formation_name, expected_depth in regional_tops.items():
                    distance = abs(boundary_depth - expected_depth)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = formation_name

                # Confidence based on distance to expected depth
                confidence = 1.0 / (1.0 + min_distance / 100)  # Decay over 100m

                formation_tops.append(
                    FormationTop(
                        depth=boundary_depth,
                        formation_name=best_match,
                        confidence=confidence,
                        method="regional_correlation",
                        log_character={},
                    )
                )
        else:
            # Simple sequential assignment
            for i, boundary_depth in enumerate(boundaries):
                if i < len(reference_sequence):
                    formation_name = reference_sequence[i]
                else:
                    formation_name = f"Unknown_{i+1}"

                formation_tops.append(
                    FormationTop(
                        depth=boundary_depth,
                        formation_name=formation_name,
                        confidence=0.7,  # Default confidence
                        method="sequential",
                        log_character={},
                    )
                )

        return formation_tops

    def detect_and_classify(
        self,
        data: pd.DataFrame,
        curves: Optional[List[str]] = None,
        reference_sequence: Optional[List[str]] = None,
        regional_tops: Optional[Dict[str, float]] = None,
    ) -> BoundaryDetectionResult:
        """
        Complete workflow: detect boundaries and classify formations.

        Args:
            data: DataFrame with log curves
            curves: List of curves to use
            reference_sequence: Expected formation sequence
            regional_tops: Regional formation depths for correlation

        Returns:
            BoundaryDetectionResult with all detected tops
        """
        self.processing_log = []

        # Step 1: Compute boundary scores
        scores = self.compute_boundary_score(data, curves)

        # Step 2: Detect boundaries
        boundaries = self.detect_boundaries(data, curves)

        # Step 3: Classify with ML if trained
        if self.boundary_classifier is not None:
            validated = self.classify_boundaries(data, boundaries, curves)
            boundaries = [depth for depth, _ in validated]
            confidences = {depth: conf for depth, conf in validated}
        else:
            confidences = {depth: 0.7 for depth in boundaries}

        # Step 4: Correlate with stratigraphy
        if reference_sequence:
            formation_tops = self.correlate_with_stratigraphy(
                boundaries,
                reference_sequence,
                regional_tops,
            )
            # Update confidences from ML
            for top in formation_tops:
                if top.depth in confidences:
                    top.confidence = min(top.confidence, confidences[top.depth])
        else:
            # Create generic tops
            formation_tops = [
                FormationTop(
                    depth=depth,
                    formation_name=f"Formation_{i+1}",
                    confidence=confidences.get(depth, 0.7),
                    method="signal_processing",
                    log_character={},
                )
                for i, depth in enumerate(boundaries)
            ]

        return BoundaryDetectionResult(
            detected_boundaries=boundaries,
            boundary_scores=scores,
            recommended_tops=formation_tops,
            processing_log=self.processing_log.copy(),
        )


def compare_tops_with_reference(
    predicted_tops: List[FormationTop],
    reference_tops: List[FormationTop],
    tolerance: float = 5.0,
) -> Dict[str, float]:
    """
    Compare predicted formation tops with reference picks.

    Args:
        predicted_tops: List of predicted FormationTop objects
        reference_tops: List of reference FormationTop objects
        tolerance: Maximum depth difference to consider a match (meters)

    Returns:
        Dictionary with accuracy metrics
    """
    # Match by formation name
    matched = 0
    total_reference = len(reference_tops)
    depth_errors = []

    for ref_top in reference_tops:
        # Find matching formation in predictions
        matching_pred = [
            p for p in predicted_tops if p.formation_name == ref_top.formation_name
        ]

        if matching_pred:
            # Use closest match
            pred_top = min(matching_pred, key=lambda p: abs(p.depth - ref_top.depth))
            depth_error = abs(pred_top.depth - ref_top.depth)

            if depth_error <= tolerance:
                matched += 1

            depth_errors.append(depth_error)

    accuracy = matched / total_reference if total_reference > 0 else 0.0
    mean_error = np.mean(depth_errors) if depth_errors else 0.0
    std_error = np.std(depth_errors) if depth_errors else 0.0

    return {
        "accuracy": accuracy,
        "mean_depth_error": mean_error,
        "std_depth_error": std_error,
        "matched": matched,
        "total_reference": total_reference,
        "false_positives": len(predicted_tops) - matched,
    }
