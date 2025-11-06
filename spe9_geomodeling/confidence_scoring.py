"""
Confidence Scoring Module

Uncertainty quantification and prediction confidence scoring for well log interpretation.
Helps identify predictions that need expert review in human-in-the-loop workflows.

Implements confidence assessment strategies for automated well log interpretation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .exceptions import DataValidationError


@dataclass
class ConfidenceScore:
    """Confidence assessment for a single prediction."""

    prediction: int  # Predicted class
    confidence: float  # 0-1, higher = more confident
    entropy_score: float  # Prediction entropy
    margin_score: float  # Margin between top 2 classes
    probability_distribution: np.ndarray  # Full probability vector
    confidence_level: str  # 'high', 'medium', 'low'
    needs_review: bool  # Flag for expert review

    def __str__(self) -> str:
        return (
            f"Prediction: {self.prediction} "
            f"(confidence: {self.confidence:.2f}, level: {self.confidence_level})"
        )


@dataclass
class WellConfidenceReport:
    """Confidence report for entire well."""

    well_name: str
    depths: np.ndarray
    predictions: np.ndarray
    confidence_scores: list[ConfidenceScore]
    overall_confidence: float
    high_confidence_fraction: float
    low_confidence_depths: list[float]  # Depths needing review
    summary: dict[str, float]

    def __str__(self) -> str:
        return (
            f"Confidence Report: {self.well_name}\n"
            f"  Overall confidence: {self.overall_confidence:.2f}\n"
            f"  High confidence: {self.high_confidence_fraction:.1%}\n"
            f"  Needs review: {len(self.low_confidence_depths)} intervals"
        )


class ConfidenceScorer:
    """
    Assess prediction confidence for well log interpretations.

    Provides multiple confidence metrics and flags uncertain predictions
    for expert review, optimizing human-in-the-loop workflows.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.8,
        low_confidence_threshold: float = 0.5,
    ):
        """
        Initialize confidence scorer.

        Args:
            high_confidence_threshold: Minimum confidence for 'high' level
            low_confidence_threshold: Below this triggers expert review
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold

    def compute_confidence_score(
        self,
        probabilities: np.ndarray,
        method: str = "composite",
    ) -> float:
        """
        Compute confidence score from probability distribution.

        Args:
            probabilities: Class probabilities (length = n_classes)
            method: 'max_prob', 'margin', 'entropy', or 'composite'

        Returns:
            Confidence score (0-1, higher = more confident)
        """
        if method == "max_prob":
            # Maximum probability
            return np.max(probabilities)

        elif method == "margin":
            # Margin between top 2 probabilities
            sorted_probs = np.sort(probabilities)
            if len(sorted_probs) < 2:
                return sorted_probs[-1]
            margin = sorted_probs[-1] - sorted_probs[-2]
            return margin

        elif method == "entropy":
            # Inverse of normalized entropy
            eps = 1e-10
            ent = entropy(probabilities + eps, base=2)
            max_entropy = np.log2(len(probabilities))
            normalized_entropy = ent / max_entropy if max_entropy > 0 else 0
            return 1 - normalized_entropy

        elif method == "composite":
            # Weighted combination of methods
            max_prob = np.max(probabilities)

            sorted_probs = np.sort(probabilities)
            margin = (
                (sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0
            )

            eps = 1e-10
            ent = entropy(probabilities + eps, base=2)
            max_entropy = np.log2(len(probabilities))
            entropy_conf = 1 - (ent / max_entropy) if max_entropy > 0 else 1.0

            # Weighted average
            composite = 0.5 * max_prob + 0.3 * margin + 0.2 * entropy_conf
            return composite

        else:
            raise DataValidationError(
                f"Unknown confidence method: {method}",
                "Valid methods: 'max_prob', 'margin', 'entropy', 'composite'",
            )

    def score_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[ConfidenceScore]:
        """
        Score confidence for each prediction.

        Args:
            predictions: Predicted class labels
            probabilities: Class probability matrix (n_samples, n_classes)

        Returns:
            List of ConfidenceScore objects
        """
        if len(predictions) != len(probabilities):
            raise DataValidationError(
                "Predictions and probabilities must have same length",
                f"Got {len(predictions)} predictions, {len(probabilities)} probability rows",
            )

        scores = []

        for pred, probs in zip(predictions, probabilities):
            # Compute confidence metrics
            confidence = self.compute_confidence_score(probs, method="composite")

            sorted_probs = np.sort(probs)
            margin = (
                (sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0
            )

            eps = 1e-10
            ent = entropy(probs + eps, base=2)

            # Determine confidence level
            if confidence >= self.high_threshold:
                confidence_level = "high"
                needs_review = False
            elif confidence >= self.low_threshold:
                confidence_level = "medium"
                needs_review = False
            else:
                confidence_level = "low"
                needs_review = True

            scores.append(
                ConfidenceScore(
                    prediction=int(pred),
                    confidence=confidence,
                    entropy_score=ent,
                    margin_score=margin,
                    probability_distribution=probs,
                    confidence_level=confidence_level,
                    needs_review=needs_review,
                )
            )

        return scores

    def create_well_report(
        self,
        well_name: str,
        depths: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> WellConfidenceReport:
        """
        Create comprehensive confidence report for a well.

        Args:
            well_name: Well identifier
            depths: Depth array
            predictions: Predicted facies/formation labels
            probabilities: Class probability matrix

        Returns:
            WellConfidenceReport with detailed confidence assessment
        """
        # Score all predictions
        confidence_scores = self.score_predictions(predictions, probabilities)

        # Compute summary statistics
        confidences = np.array([s.confidence for s in confidence_scores])
        overall_confidence = confidences.mean()

        high_conf_count = sum(
            1 for s in confidence_scores if s.confidence_level == "high"
        )
        high_confidence_fraction = high_conf_count / len(confidence_scores)

        # Identify depths needing review
        low_confidence_depths = [
            depth
            for depth, score in zip(depths, confidence_scores)
            if score.needs_review
        ]

        # Create summary dictionary
        summary = {
            "mean_confidence": overall_confidence,
            "median_confidence": np.median(confidences),
            "min_confidence": confidences.min(),
            "max_confidence": confidences.max(),
            "std_confidence": confidences.std(),
            "high_confidence_fraction": high_confidence_fraction,
            "medium_confidence_fraction": sum(
                1 for s in confidence_scores if s.confidence_level == "medium"
            )
            / len(confidence_scores),
            "low_confidence_fraction": sum(
                1 for s in confidence_scores if s.confidence_level == "low"
            )
            / len(confidence_scores),
            "needs_review_fraction": len(low_confidence_depths) / len(depths),
        }

        return WellConfidenceReport(
            well_name=well_name,
            depths=depths,
            predictions=predictions,
            confidence_scores=confidence_scores,
            overall_confidence=overall_confidence,
            high_confidence_fraction=high_confidence_fraction,
            low_confidence_depths=low_confidence_depths,
            summary=summary,
        )

    def triage_predictions(
        self,
        reports: list[WellConfidenceReport],
        review_budget: int,
    ) -> list[tuple[str, float]]:
        """
        Triage predictions across multiple wells for expert review.

        Prioritizes the most uncertain predictions to maximize value
        of limited expert review time.

        Args:
            reports: List of WellConfidenceReport objects
            review_budget: Maximum number of intervals to flag for review

        Returns:
            List of (well_name, depth) tuples ranked by uncertainty
        """
        # Collect all low-confidence predictions with their details
        candidates = []

        for report in reports:
            for depth, score in zip(report.depths, report.confidence_scores):
                if score.needs_review:
                    # Use inverse confidence as uncertainty score
                    uncertainty = 1 - score.confidence
                    candidates.append((report.well_name, float(depth), uncertainty))

        # Sort by uncertainty (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Return top candidates up to budget
        prioritized = [(well, depth) for well, depth, _ in candidates[:review_budget]]

        return prioritized

    def confidence_by_depth_interval(
        self,
        report: WellConfidenceReport,
        interval_thickness: float = 10.0,
    ) -> pd.DataFrame:
        """
        Aggregate confidence scores by depth intervals.

        Useful for identifying problematic zones in the well.

        Args:
            report: WellConfidenceReport
            interval_thickness: Thickness of depth bins (meters)

        Returns:
            DataFrame with interval statistics
        """
        # Create depth bins
        min_depth = report.depths.min()
        max_depth = report.depths.max()
        bins = np.arange(min_depth, max_depth + interval_thickness, interval_thickness)

        # Assign each depth to a bin
        bin_indices = np.digitize(report.depths, bins) - 1

        # Aggregate by bin
        intervals = []

        for bin_idx in range(len(bins) - 1):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue

            interval_scores = [s for s, m in zip(report.confidence_scores, mask) if m]
            interval_confidences = [s.confidence for s in interval_scores]

            intervals.append(
                {
                    "depth_top": bins[bin_idx],
                    "depth_bottom": bins[bin_idx + 1],
                    "mean_confidence": np.mean(interval_confidences),
                    "min_confidence": np.min(interval_confidences),
                    "n_samples": mask.sum(),
                    "needs_review_count": sum(
                        1 for s in interval_scores if s.needs_review
                    ),
                    "dominant_prediction": (
                        pd.Series([s.prediction for s in interval_scores]).mode()[0]
                        if interval_scores
                        else -1
                    ),
                }
            )

        return pd.DataFrame(intervals)

    def confidence_by_facies(
        self,
        report: WellConfidenceReport,
        facies_names: Optional[dict[int, str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze confidence by predicted facies type.

        Identifies which facies are harder to classify accurately.

        Args:
            report: WellConfidenceReport
            facies_names: Optional mapping of facies codes to names

        Returns:
            DataFrame with per-facies confidence statistics
        """
        # Group by predicted facies
        facies_stats = {}

        for pred, score in zip(report.predictions, report.confidence_scores):
            if pred not in facies_stats:
                facies_stats[pred] = {
                    "confidences": [],
                    "count": 0,
                    "needs_review": 0,
                }

            facies_stats[pred]["confidences"].append(score.confidence)
            facies_stats[pred]["count"] += 1
            if score.needs_review:
                facies_stats[pred]["needs_review"] += 1

        # Create summary
        rows = []
        for facies_code, stats in facies_stats.items():
            confidences = stats["confidences"]
            rows.append(
                {
                    "facies_code": facies_code,
                    "facies_name": (
                        facies_names.get(facies_code, f"Facies_{facies_code}")
                        if facies_names
                        else f"Facies_{facies_code}"
                    ),
                    "count": stats["count"],
                    "mean_confidence": np.mean(confidences),
                    "std_confidence": np.std(confidences),
                    "min_confidence": np.min(confidences),
                    "needs_review_count": stats["needs_review"],
                    "needs_review_fraction": stats["needs_review"] / stats["count"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_confidence")

        return df


def compare_confidence_across_wells(
    reports: list[WellConfidenceReport],
) -> pd.DataFrame:
    """
    Compare confidence metrics across multiple wells.

    Args:
        reports: List of WellConfidenceReport objects

    Returns:
        DataFrame with per-well confidence comparison
    """
    rows = []

    for report in reports:
        rows.append(
            {
                "well_name": report.well_name,
                "n_samples": len(report.predictions),
                "overall_confidence": report.overall_confidence,
                "high_confidence_fraction": report.high_confidence_fraction,
                "needs_review_count": len(report.low_confidence_depths),
                "needs_review_fraction": report.summary["needs_review_fraction"],
                "mean_confidence": report.summary["mean_confidence"],
                "std_confidence": report.summary["std_confidence"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("overall_confidence", ascending=False)

    return df


def export_review_list(
    reports: list[WellConfidenceReport],
    output_file: str,
    max_items: Optional[int] = None,
):
    """
    Export prioritized review list for expert QC.

    Args:
        reports: List of WellConfidenceReport objects
        output_file: Path to output CSV file
        max_items: Maximum number of items to export (None = all)
    """
    # Collect all items needing review
    review_items = []

    for report in reports:
        for depth, score in zip(report.depths, report.confidence_scores):
            if score.needs_review:
                review_items.append(
                    {
                        "well_name": report.well_name,
                        "depth": depth,
                        "prediction": score.prediction,
                        "confidence": score.confidence,
                        "entropy": score.entropy_score,
                        "margin": score.margin_score,
                        "uncertainty": 1 - score.confidence,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(review_items)

    # Sort by uncertainty (descending)
    df = df.sort_values("uncertainty", ascending=False)

    # Limit if requested
    if max_items:
        df = df.head(max_items)

    # Export
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} items for review to {output_file}")
