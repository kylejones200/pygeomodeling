"""
Workflow Manager Module

Manages human-in-the-loop iterative refinement for well log interpretation.
Tracks model versions, corrections, and continuous improvement cycles.

Implements the workflow orchestration described in automated well log
interpretation systems.
"""

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class WorkflowIteration:
    """Represents a single iteration of the workflow."""

    iteration_number: int
    date: str
    model_version: str
    wells_processed: list[str]
    corrections_received: int
    model_performance: dict[str, float]
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "iteration_number": self.iteration_number,
            "date": self.date,
            "model_version": self.model_version,
            "wells_processed": self.wells_processed,
            "corrections_received": self.corrections_received,
            "model_performance": self.model_performance,
            "notes": self.notes,
        }


@dataclass
class CorrectionRecord:
    """Record of an expert correction."""

    well_name: str
    depth: float
    original_prediction: int
    corrected_label: int
    confidence_score: float
    correction_date: str
    expert_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "well_name": self.well_name,
            "depth": self.depth,
            "original_prediction": int(self.original_prediction),
            "corrected_label": int(self.corrected_label),
            "confidence_score": float(self.confidence_score),
            "correction_date": self.correction_date,
            "expert_id": self.expert_id,
            "notes": self.notes,
        }


@dataclass
class WorkflowState:
    """Current state of the workflow."""

    current_iteration: int
    total_wells: int
    interpreted_wells: int
    reviewed_wells: int
    total_corrections: int
    current_model_version: str
    iterations: list[WorkflowIteration] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_iteration": self.current_iteration,
            "total_wells": self.total_wells,
            "interpreted_wells": self.interpreted_wells,
            "reviewed_wells": self.reviewed_wells,
            "total_corrections": self.total_corrections,
            "current_model_version": self.current_model_version,
            "iterations": [it.to_dict() for it in self.iterations],
            "corrections": [corr.to_dict() for corr in self.corrections],
        }

    @staticmethod
    def from_dict(data: dict) -> "WorkflowState":
        """Load from dictionary."""
        state = WorkflowState(
            current_iteration=data["current_iteration"],
            total_wells=data["total_wells"],
            interpreted_wells=data["interpreted_wells"],
            reviewed_wells=data["reviewed_wells"],
            total_corrections=data["total_corrections"],
            current_model_version=data["current_model_version"],
        )

        # Reconstruct iterations
        for it_data in data.get("iterations", []):
            state.iterations.append(WorkflowIteration(**it_data))

        # Reconstruct corrections
        for corr_data in data.get("corrections", []):
            state.corrections.append(CorrectionRecord(**corr_data))

        return state


class WorkflowManager:
    """
    Manages iterative refinement workflow for well log interpretation.

    Orchestrates the complete cycle:
    1. Initial model training
    2. Batch prediction
    3. Expert review and correction
    4. Model retraining
    5. Iteration
    """

    def __init__(self, project_dir: str):
        """
        Initialize workflow manager.

        Args:
            project_dir: Base directory for workflow state and data
        """
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.project_dir / "workflow_state.json"
        self.corrections_dir = self.project_dir / "corrections"
        self.corrections_dir.mkdir(exist_ok=True)

        self.models_dir = self.project_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.reports_dir = self.project_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Load or initialize state
        if self.state_file.exists():
            self.state = self.load_state()
        else:
            self.state = WorkflowState(
                current_iteration=0,
                total_wells=0,
                interpreted_wells=0,
                reviewed_wells=0,
                total_corrections=0,
                current_model_version="v0.0",
            )
            self.save_state()

    def save_state(self):
        """Save current workflow state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load_state(self) -> WorkflowState:
        """Load workflow state from disk."""
        with open(self.state_file) as f:
            data = json.load(f)
        return WorkflowState.from_dict(data)

    def start_new_iteration(
        self,
        wells_to_process: list[str],
        model_version: Optional[str] = None,
    ):
        """
        Start a new workflow iteration.

        Args:
            wells_to_process: List of well names to interpret
            model_version: Optional model version identifier
        """
        self.state.current_iteration += 1

        if model_version is None:
            self.state.current_model_version = f"v{self.state.current_iteration}.0"
        else:
            self.state.current_model_version = model_version

        print(f"Starting iteration {self.state.current_iteration}")
        print(f"  Model version: {self.state.current_model_version}")
        print(f"  Wells to process: {len(wells_to_process)}")

        self.state.total_wells = len(wells_to_process)
        self.save_state()

    def record_well_interpreted(
        self,
        well_name: str,
        prediction_file: str,
    ):
        """
        Record that a well has been interpreted.

        Args:
            well_name: Well identifier
            prediction_file: Path to prediction results
        """
        self.state.interpreted_wells += 1
        print(
            f"Interpreted well: {well_name} ({self.state.interpreted_wells}/{self.state.total_wells})"
        )
        self.save_state()

    def import_corrections(
        self,
        correction_file: str,
        expert_id: Optional[str] = None,
    ) -> int:
        """
        Import expert corrections from review file.

        Args:
            correction_file: Path to correction CSV
            expert_id: Optional expert identifier

        Returns:
            Number of corrections imported
        """
        df = pd.read_csv(correction_file)

        # Filter to corrected samples
        if "Corrected" in df.columns:
            corrected = df[df["Corrected"] == True]
        else:
            # Assume all rows are corrections if no flag column
            corrected = df[df["ML_Prediction"] != df["Expert_Correction"]]

        if len(corrected) == 0:
            print("No corrections found in file")
            return 0

        correction_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Record each correction
        for _, row in corrected.iterrows():
            correction = CorrectionRecord(
                well_name=row["Well"],
                depth=row["Depth_m"],
                original_prediction=int(row["ML_Prediction"]),
                corrected_label=int(row["Expert_Correction"]),
                confidence_score=row["Confidence"],
                correction_date=correction_date,
                expert_id=expert_id,
                notes=row.get("Notes", ""),
            )
            self.state.corrections.append(correction)

        self.state.total_corrections += len(corrected)
        self.state.reviewed_wells += len(corrected["Well"].unique())

        print(f"Imported {len(corrected)} corrections from {correction_file}")
        print(f"  Total corrections: {self.state.total_corrections}")
        print(f"  Reviewed wells: {self.state.reviewed_wells}")

        self.save_state()

        return len(corrected)

    def get_training_data_with_corrections(
        self,
        original_training_data: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "Facies",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get augmented training data including all corrections.

        Args:
            original_training_data: Initial training dataset
            feature_columns: List of feature column names
            label_column: Name of label column

        Returns:
            X_augmented, y_augmented with corrections integrated
        """
        X_original = original_training_data[feature_columns]
        y_original = original_training_data[label_column]

        if not self.state.corrections:
            return X_original, y_original

        # Convert corrections to DataFrame
        correction_rows = []
        for corr in self.state.corrections:
            # This is simplified - in practice you'd need to look up
            # the feature values at the corrected depth
            correction_rows.append(
                {
                    "well": corr.well_name,
                    "depth": corr.depth,
                    label_column: corr.corrected_label,
                }
            )

        corrections_df = pd.DataFrame(correction_rows)

        # Note: You would need to merge with actual feature data here
        # This is a placeholder showing the pattern

        print(
            f"Training with {len(X_original)} original + {len(corrections_df)} corrected samples"
        )

        return X_original, y_original

    def complete_iteration(
        self,
        model_performance: dict[str, float],
        notes: str = "",
    ):
        """
        Complete current iteration and record results.

        Args:
            model_performance: Dictionary of performance metrics
            notes: Optional notes about the iteration
        """
        iteration = WorkflowIteration(
            iteration_number=self.state.current_iteration,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            model_version=self.state.current_model_version,
            wells_processed=[f"well_{i}" for i in range(self.state.interpreted_wells)],
            corrections_received=len(
                [
                    c
                    for c in self.state.corrections
                    if c.correction_date == datetime.datetime.now().strftime("%Y-%m-%d")
                ]
            ),
            model_performance=model_performance,
            notes=notes,
        )

        self.state.iterations.append(iteration)

        print(f"\nCompleted iteration {self.state.current_iteration}")
        print(f"  Wells processed: {self.state.interpreted_wells}")
        print(f"  Corrections received: {iteration.corrections_received}")
        print(f"  Model performance: {model_performance}")

        self.save_state()
        self.generate_progress_report()

    def generate_progress_report(self) -> pd.DataFrame:
        """
        Generate progress report across all iterations.

        Returns:
            DataFrame with iteration history
        """
        if not self.state.iterations:
            print("No iterations completed yet")
            return pd.DataFrame()

        rows = []
        for iteration in self.state.iterations:
            row = {
                "Iteration": iteration.iteration_number,
                "Date": iteration.date,
                "Model_Version": iteration.model_version,
                "Wells_Processed": len(iteration.wells_processed),
                "Corrections": iteration.corrections_received,
                **iteration.model_performance,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save report
        report_file = self.reports_dir / "progress_report.csv"
        df.to_csv(report_file, index=False)

        print(f"\nProgress Report:")
        print(df.to_string())
        print(f"\nFull report saved to: {report_file}")

        return df

    def get_correction_statistics(self) -> dict:
        """
        Analyze correction patterns.

        Returns:
            Dictionary with correction statistics
        """
        if not self.state.corrections:
            return {}

        corrections_df = pd.DataFrame([c.to_dict() for c in self.state.corrections])

        stats = {
            "total_corrections": len(corrections_df),
            "unique_wells": corrections_df["well_name"].nunique(),
            "avg_confidence": corrections_df["confidence_score"].mean(),
            "corrections_by_well": corrections_df.groupby("well_name").size().to_dict(),
            "most_confused_pairs": self._get_confusion_pairs(corrections_df),
        }

        return stats

    def _get_confusion_pairs(
        self, corrections_df: pd.DataFrame
    ) -> list[tuple[int, int, int]]:
        """Get most common prediction-correction pairs."""
        pairs = corrections_df.groupby(
            ["original_prediction", "corrected_label"]
        ).size()
        pairs = pairs.sort_values(ascending=False).head(5)

        return [
            (int(orig), int(corr), int(count)) for (orig, corr), count in pairs.items()
        ]

    def export_workflow_summary(self, output_file: Optional[str] = None):
        """
        Export comprehensive workflow summary.

        Args:
            output_file: Optional output path (default: reports/workflow_summary.txt)
        """
        if output_file is None:
            output_file = str(self.reports_dir / "workflow_summary.txt")

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("WORKFLOW SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Current Status:\n")
            f.write(f"  Iteration: {self.state.current_iteration}\n")
            f.write(f"  Model Version: {self.state.current_model_version}\n")
            f.write(f"  Total Wells: {self.state.total_wells}\n")
            f.write(f"  Interpreted: {self.state.interpreted_wells}\n")
            f.write(f"  Reviewed: {self.state.reviewed_wells}\n")
            f.write(f"  Total Corrections: {self.state.total_corrections}\n\n")

            f.write("Iteration History:\n")
            f.write("-" * 80 + "\n")
            for iteration in self.state.iterations:
                f.write(
                    f"\nIteration {iteration.iteration_number} ({iteration.date}):\n"
                )
                f.write(f"  Model: {iteration.model_version}\n")
                f.write(f"  Wells: {len(iteration.wells_processed)}\n")
                f.write(f"  Corrections: {iteration.corrections_received}\n")
                f.write(f"  Performance: {iteration.model_performance}\n")
                if iteration.notes:
                    f.write(f"  Notes: {iteration.notes}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(
                f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        print(f"Workflow summary exported to {output_file}")


def create_workflow_dashboard(workflow_manager: WorkflowManager) -> dict:
    """
    Create dashboard data for workflow monitoring.

    Args:
        workflow_manager: WorkflowManager instance

    Returns:
        Dictionary with dashboard metrics
    """
    state = workflow_manager.state

    # Calculate key metrics
    review_rate = (
        state.reviewed_wells / state.interpreted_wells
        if state.interpreted_wells > 0
        else 0
    )
    correction_rate = (
        state.total_corrections / state.interpreted_wells
        if state.interpreted_wells > 0
        else 0
    )

    # Get performance trend
    performance_trend = []
    if state.iterations:
        for iteration in state.iterations:
            if "accuracy" in iteration.model_performance:
                performance_trend.append(iteration.model_performance["accuracy"])

    dashboard = {
        "current_iteration": state.current_iteration,
        "progress_percent": (
            (state.interpreted_wells / state.total_wells * 100)
            if state.total_wells > 0
            else 0
        ),
        "review_rate": review_rate * 100,
        "correction_rate": correction_rate * 100,
        "performance_trend": performance_trend,
        "total_corrections": state.total_corrections,
        "model_version": state.current_model_version,
    }

    return dashboard
