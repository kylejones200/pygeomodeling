"""
Well Log Processing Module

Advanced data preparation pipeline for well log interpretation automation.
Handles curve identification, normalization, depth alignment, quality control,
and missing value imputation.

Implements the data standardization approach described in automated well log
interpretation workflows.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import zscore

from .exceptions import InvalidParameterError

# Standard curve type definitions with typical statistical ranges
CURVE_SIGNATURES = {
    "GR": {
        "names": ["GR", "CGR", "GRD", "GAPI", "GR_EDTC"],
        "typical_range": (0, 200),
        "unit_patterns": ["API", "GAPI", "api"],
        "description": "Gamma Ray",
    },
    "RESISTIVITY": {
        "names": [
            "ILD",
            "ILM",
            "ILS",
            "RT",
            "RES",
            "RESD",
            "RESS",
            "LLD",
            "LLS",
            "AT90",
            "RACELM",
        ],
        "typical_range": (0.1, 2000),
        "unit_patterns": ["OHMM", "ohm.m", "ohm-m"],
        "description": "Resistivity (deep/medium/shallow)",
    },
    "DENSITY": {
        "names": ["RHOB", "RHOZ", "ZDEN", "DEN", "RHOBC"],
        "typical_range": (1.5, 3.0),
        "unit_patterns": ["G/C3", "g/cm3", "g/cc", "G/CC"],
        "description": "Bulk Density",
    },
    "NEUTRON": {
        "names": ["NPHI", "PHIN", "NEU", "TNPH", "NPOR"],
        "typical_range": (0.0, 0.6),
        "unit_patterns": ["V/V", "frac", "%", "decimal"],
        "description": "Neutron Porosity",
    },
    "SONIC": {
        "names": ["DT", "DTC", "DTCO", "AC", "DTHM", "DTS"],
        "typical_range": (40, 200),
        "unit_patterns": ["US/F", "us/ft", "μs/ft"],
        "description": "Sonic (compressional)",
    },
    "CALIPER": {
        "names": ["CALI", "CAL", "CALS", "BS"],
        "typical_range": (6, 20),
        "unit_patterns": ["IN", "in", "inches"],
        "description": "Caliper (borehole diameter)",
    },
    "PE": {
        "names": ["PE", "PEF", "PEFZ"],
        "typical_range": (1, 10),
        "unit_patterns": ["B/E", "barns/electron"],
        "description": "Photoelectric Effect",
    },
    "DEPTH": {
        "names": ["DEPT", "DEPTH", "MD", "TVDSS", "TVD"],
        "typical_range": (0, 10000),
        "unit_patterns": ["M", "FT", "m", "ft", "meters", "feet"],
        "description": "Measured/True Vertical Depth",
    },
}


@dataclass
class CurveQuality:
    """Quality assessment for a log curve."""

    curve_name: str
    data_coverage: float  # Fraction of non-null values
    outlier_fraction: float
    mean: float
    std: float
    min_value: float
    max_value: float
    quality_flag: str  # 'good', 'acceptable', 'poor'
    issues: list[str]  # Specific quality problems

    def __str__(self) -> str:
        return (
            f"{self.curve_name}: {self.quality_flag.upper()}\n"
            f"  Coverage: {self.data_coverage:.1%}\n"
            f"  Range: [{self.min_value:.2f}, {self.max_value:.2f}]\n"
            f"  Issues: {', '.join(self.issues) if self.issues else 'None'}"
        )


@dataclass
class ProcessedWellLogs:
    """Container for processed well log data."""

    data: pd.DataFrame
    curve_types: dict[str, str]  # curve_name -> curve_type
    quality_report: dict[str, CurveQuality]
    processing_log: list[str]
    original_curves: set[str]
    added_curves: set[str]

    def __str__(self) -> str:
        good_curves = sum(
            1 for q in self.quality_report.values() if q.quality_flag == "good"
        )
        return (
            f"Processed Well Logs:\n"
            f"  Depth range: {self.data.index.min():.1f} - {self.data.index.max():.1f}\n"
            f"  Curves: {len(self.curve_types)} total ({good_curves} good quality)\n"
            f"  Processing steps: {len(self.processing_log)}"
        )


class WellLogProcessor:
    """
    Advanced well log processing pipeline.

    Automates curve identification, normalization, quality control, and preparation
    for machine learning workflows.
    """

    def __init__(
        self,
        null_value: float = -999.25,
        min_coverage: float = 0.5,
        outlier_threshold: float = 5.0,
    ):
        """
        Initialize well log processor.

        Args:
            null_value: Value indicating missing data
            min_coverage: Minimum fraction of valid data for acceptable quality
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.null_value = null_value
        self.min_coverage = min_coverage
        self.outlier_threshold = outlier_threshold
        self.processing_log = []

    def identify_curve_type(
        self,
        curve_name: str,
        data: pd.Series,
        unit: Optional[str] = None,
    ) -> Optional[str]:
        """
        Automatically identify curve type from name, unit, and statistics.

        Args:
            curve_name: Curve mnemonic
            data: Curve values
            unit: Unit string (if available)

        Returns:
            Identified curve type or None if unknown
        """
        curve_name_upper = curve_name.upper()

        # First try name matching
        for curve_type, signature in CURVE_SIGNATURES.items():
            if any(name in curve_name_upper for name in signature["names"]):
                # Verify with statistical check
                if self._verify_curve_statistics(data, signature["typical_range"]):
                    return curve_type

        # Try unit matching if available
        if unit:
            unit_upper = unit.upper()
            for curve_type, signature in CURVE_SIGNATURES.items():
                if any(
                    pattern.upper() in unit_upper
                    for pattern in signature["unit_patterns"]
                ):
                    if self._verify_curve_statistics(data, signature["typical_range"]):
                        return curve_type

        # Try statistical pattern matching
        valid_data = data[data != self.null_value].dropna()
        if len(valid_data) < 10:
            return None

        data_range = (valid_data.min(), valid_data.max())
        best_match = None
        best_overlap = 0

        for curve_type, signature in CURVE_SIGNATURES.items():
            overlap = self._range_overlap(data_range, signature["typical_range"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = curve_type

        # Only return if confidence is high enough
        return best_match if best_overlap > 0.5 else None

    def _verify_curve_statistics(
        self,
        data: pd.Series,
        expected_range: tuple[float, float],
    ) -> bool:
        """Check if curve statistics match expected range."""
        valid_data = data[data != self.null_value].dropna()
        if len(valid_data) < 10:
            return False

        median = valid_data.median()
        min_val, max_val = expected_range

        # Check if median falls within or near expected range
        # Allow 50% margin for different basins/formations
        margin = (max_val - min_val) * 0.5
        return (min_val - margin) <= median <= (max_val + margin)

    def _range_overlap(
        self,
        range1: tuple[float, float],
        range2: tuple[float, float],
    ) -> float:
        """Calculate overlap coefficient between two ranges."""
        min1, max1 = range1
        min2, max2 = range2

        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)

        if overlap_max <= overlap_min:
            return 0.0

        overlap_range = overlap_max - overlap_min
        range1_size = max1 - min1
        range2_size = max2 - min2

        # Return overlap as fraction of smaller range
        smaller_range = min(range1_size, range2_size)
        return overlap_range / smaller_range if smaller_range > 0 else 0.0

    def normalize_curve_names(
        self,
        data: pd.DataFrame,
        curve_types: dict[str, str],
    ) -> pd.DataFrame:
        """
        Standardize curve names to common mnemonics.

        Args:
            data: DataFrame with original curve names
            curve_types: Mapping of curve names to types

        Returns:
            DataFrame with standardized names
        """
        name_mapping = {}

        for curve_name, curve_type in curve_types.items():
            if curve_type in CURVE_SIGNATURES:
                # Use first standard name as the canonical name
                standard_name = CURVE_SIGNATURES[curve_type]["names"][0]
                if curve_name != standard_name:
                    name_mapping[curve_name] = standard_name

        if name_mapping:
            data = data.rename(columns=name_mapping)
            self.processing_log.append(f"Normalized {len(name_mapping)} curve names")

        return data

    def align_depth(
        self,
        data: pd.DataFrame,
        depth_col: str = "DEPT",
        target_step: Optional[float] = None,
        datum_shift: float = 0.0,
    ) -> pd.DataFrame:
        """
        Align depth index to uniform spacing and common datum.

        Args:
            data: DataFrame with depth index or column
            depth_col: Name of depth column if not index
            target_step: Target depth spacing (None = auto-detect)
            datum_shift: Vertical shift to apply (e.g., to subsea)

        Returns:
            DataFrame with aligned depth index
        """
        # Ensure depth is the index
        if depth_col in data.columns:
            data = data.set_index(depth_col)

        original_index = data.index.values

        # Auto-detect step if not provided
        if target_step is None:
            steps = np.diff(original_index)
            target_step = np.median(steps[steps > 0])
            self.processing_log.append(f"Auto-detected depth step: {target_step:.4f}")

        # Create uniform depth grid
        min_depth = original_index.min() + datum_shift
        max_depth = original_index.max() + datum_shift
        new_depth = np.arange(min_depth, max_depth + target_step, target_step)

        # Interpolate all curves to new depth grid
        aligned_data = pd.DataFrame(index=new_depth)

        for col in data.columns:
            # Remove null values for interpolation
            valid_mask = data[col] != self.null_value
            if valid_mask.sum() < 2:
                aligned_data[col] = self.null_value
                continue

            valid_depth = original_index[valid_mask]
            valid_values = data.loc[valid_mask, col].values

            # Interpolate with extrapolation handling
            interp_func = interpolate.interp1d(
                valid_depth,
                valid_values,
                kind="linear",
                bounds_error=False,
                fill_value=self.null_value,
            )

            aligned_data[col] = interp_func(new_depth)

        self.processing_log.append(
            f"Aligned depth: {len(original_index)} -> {len(new_depth)} points"
        )

        return aligned_data

    def impute_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "linear",
        max_gap: int = 10,
    ) -> pd.DataFrame:
        """
        Fill missing values using various interpolation strategies.

        Args:
            data: DataFrame with missing values
            method: 'linear', 'polynomial', 'median', or 'forward'
            max_gap: Maximum number of consecutive points to interpolate

        Returns:
            DataFrame with imputed values
        """
        imputed = data.copy()

        for col in data.columns:
            # Replace null values with NaN for processing
            series = data[col].replace(self.null_value, np.nan)

            # Count missing values
            missing_count = series.isna().sum()
            if missing_count == 0:
                continue

            if method == "linear":
                # Linear interpolation with max gap limit
                series = series.interpolate(
                    method="linear",
                    limit=max_gap,
                    limit_direction="both",
                )
            elif method == "polynomial":
                series = series.interpolate(method="polynomial", order=2, limit=max_gap)
            elif method == "median":
                # Use rolling median for gaps
                series = series.fillna(
                    series.rolling(window=5, center=True, min_periods=1).median()
                )
            elif method == "forward":
                series = series.fillna(method="ffill", limit=max_gap)
            else:
                raise InvalidParameterError(
                    f"Unknown imputation method: {method}",
                    f"Valid methods: 'linear', 'polynomial', 'median', 'forward'",
                )

            imputed[col] = series.fillna(self.null_value)
            filled_count = missing_count - imputed[col].eq(self.null_value).sum()

            if filled_count > 0:
                self.processing_log.append(
                    f"Imputed {filled_count}/{missing_count} values in {col}"
                )

        return imputed

    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        threshold: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """
        Detect outliers in log curves.

        Args:
            data: DataFrame with log curves
            method: 'zscore', 'iqr', or 'median_abs'
            threshold: Detection threshold (method-specific)

        Returns:
            Dictionary mapping curve names to outlier masks
        """
        if threshold is None:
            threshold = self.outlier_threshold

        outliers = {}

        for col in data.columns:
            valid_data = data[col][data[col] != self.null_value]

            if len(valid_data) < 10:
                outliers[col] = np.zeros(len(data), dtype=bool)
                continue

            if method == "zscore":
                z_scores = np.abs(zscore(valid_data, nan_policy="omit"))
                outlier_mask = z_scores > threshold
            elif method == "iqr":
                q1, q3 = valid_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outlier_mask = (valid_data < lower) | (valid_data > upper)
            elif method == "median_abs":
                median = valid_data.median()
                mad = np.median(np.abs(valid_data - median))
                modified_z = 0.6745 * (valid_data - median) / mad if mad > 0 else 0
                outlier_mask = np.abs(modified_z) > threshold
            else:
                raise InvalidParameterError(
                    f"Unknown outlier detection method: {method}",
                    "Valid methods: 'zscore', 'iqr', 'median_abs'",
                )

            # Map back to full data index
            full_outlier_mask = np.zeros(len(data), dtype=bool)
            full_outlier_mask[data[col] != self.null_value] = outlier_mask
            outliers[col] = full_outlier_mask

        return outliers

    def assess_quality(
        self,
        data: pd.DataFrame,
        outliers: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, CurveQuality]:
        """
        Assess quality of each log curve.

        Args:
            data: DataFrame with log curves
            outliers: Optional outlier masks from detect_outliers

        Returns:
            Dictionary of quality assessments
        """
        if outliers is None:
            outliers = self.detect_outliers(data)

        quality_report = {}

        for col in data.columns:
            valid_mask = data[col] != self.null_value
            valid_data = data[col][valid_mask]

            if len(valid_data) == 0:
                quality_report[col] = CurveQuality(
                    curve_name=col,
                    data_coverage=0.0,
                    outlier_fraction=0.0,
                    mean=np.nan,
                    std=np.nan,
                    min_value=np.nan,
                    max_value=np.nan,
                    quality_flag="poor",
                    issues=["No valid data"],
                )
                continue

            coverage = len(valid_data) / len(data)
            outlier_frac = outliers[col].sum() / len(data)

            issues = []
            if coverage < self.min_coverage:
                issues.append(f"Low coverage ({coverage:.1%})")
            if outlier_frac > 0.05:
                issues.append(f"High outlier rate ({outlier_frac:.1%})")

            # Determine quality flag
            if coverage >= 0.8 and outlier_frac < 0.03:
                quality_flag = "good"
            elif coverage >= self.min_coverage and outlier_frac < 0.10:
                quality_flag = "acceptable"
            else:
                quality_flag = "poor"

            quality_report[col] = CurveQuality(
                curve_name=col,
                data_coverage=coverage,
                outlier_fraction=outlier_frac,
                mean=float(valid_data.mean()),
                std=float(valid_data.std()),
                min_value=float(valid_data.min()),
                max_value=float(valid_data.max()),
                quality_flag=quality_flag,
                issues=issues,
            )

        return quality_report

    def process_well_logs(
        self,
        data: pd.DataFrame,
        curve_info: Optional[dict[str, dict[str, str]]] = None,
        normalize_names: bool = True,
        align_depth_grid: bool = True,
        impute_missing: bool = True,
        target_depth_step: Optional[float] = None,
    ) -> ProcessedWellLogs:
        """
        Complete processing pipeline for well logs.

        Args:
            data: Raw log data
            curve_info: Optional dict with curve metadata (unit, description)
            normalize_names: Standardize curve names
            align_depth_grid: Align to uniform depth spacing
            impute_missing: Fill missing values
            target_depth_step: Target depth spacing (None = auto-detect)

        Returns:
            ProcessedWellLogs with cleaned data and quality report
        """
        self.processing_log = []
        original_curves = set(data.columns)

        # Step 1: Identify curve types
        self.processing_log.append("Starting well log processing")
        curve_types = {}

        for col in data.columns:
            unit = None
            if curve_info and col in curve_info:
                unit = curve_info[col].get("unit")

            curve_type = self.identify_curve_type(col, data[col], unit)
            if curve_type:
                curve_types[col] = curve_type

        self.processing_log.append(f"Identified {len(curve_types)} curve types")

        # Step 2: Normalize curve names
        processed_data = data.copy()
        if normalize_names and curve_types:
            processed_data = self.normalize_curve_names(processed_data, curve_types)
            # Update curve_types with new names
            new_curve_types = {}
            for old_name, curve_type in curve_types.items():
                for new_name in processed_data.columns:
                    if new_name == old_name or (
                        curve_type in CURVE_SIGNATURES
                        and new_name in CURVE_SIGNATURES[curve_type]["names"]
                    ):
                        new_curve_types[new_name] = curve_type
                        break
            curve_types = new_curve_types

        # Step 3: Align depth grid
        if align_depth_grid:
            processed_data = self.align_depth(
                processed_data,
                target_step=target_depth_step,
            )

        # Step 4: Impute missing values
        if impute_missing:
            processed_data = self.impute_missing_values(processed_data, method="linear")

        # Step 5: Detect outliers and assess quality
        outliers = self.detect_outliers(processed_data)
        quality_report = self.assess_quality(processed_data, outliers)

        # Log quality summary
        good_count = sum(1 for q in quality_report.values() if q.quality_flag == "good")
        acceptable_count = sum(
            1 for q in quality_report.values() if q.quality_flag == "acceptable"
        )
        poor_count = sum(1 for q in quality_report.values() if q.quality_flag == "poor")

        self.processing_log.append(
            f"Quality: {good_count} good, {acceptable_count} acceptable, {poor_count} poor"
        )

        added_curves = set(processed_data.columns) - original_curves

        return ProcessedWellLogs(
            data=processed_data,
            curve_types=curve_types,
            quality_report=quality_report,
            processing_log=self.processing_log.copy(),
            original_curves=original_curves,
            added_curves=added_curves,
        )


def process_multiple_wells(
    well_data_dict: dict[str, pd.DataFrame],
    processor: Optional[WellLogProcessor] = None,
    **processing_kwargs,
) -> dict[str, ProcessedWellLogs]:
    """
    Process multiple wells with consistent parameters.

    Args:
        well_data_dict: Dictionary mapping well names to DataFrames
        processor: Optional WellLogProcessor instance
        **processing_kwargs: Arguments passed to process_well_logs

    Returns:
        Dictionary mapping well names to ProcessedWellLogs
    """
    if processor is None:
        processor = WellLogProcessor()

    processed_wells = {}

    for well_name, well_data in well_data_dict.items():
        try:
            processed = processor.process_well_logs(well_data, **processing_kwargs)
            processed_wells[well_name] = processed
        except Exception as e:
            warnings.warn(f"Failed to process well {well_name}: {e}")
            continue

    return processed_wells
