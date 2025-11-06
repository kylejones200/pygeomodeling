"""
Facies Classification Module

Machine learning methods for lithofacies classification from well log data.
Implements supervised learning algorithms (SVM, Random Forest, Neural Networks)
for identifying rock types based on wireline measurements.

Based on the SEG tutorial by Brendon Hall (Enthought) using the Hugoton and
Panoma gas fields dataset from University of Kansas.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

from .exceptions import DataValidationError, InvalidParameterError

# Standard facies descriptions from Hugoton/Panoma dataset
FACIES_LABELS = {
    1: {"name": "SS", "description": "Non-marine sandstone", "adjacent": [2]},
    2: {
        "name": "CSiS",
        "description": "Non-marine coarse siltstone",
        "adjacent": [1, 3],
    },
    3: {"name": "FSiS", "description": "Non-marine fine siltstone", "adjacent": [2]},
    4: {"name": "SiSh", "description": "Marine siltstone and shale", "adjacent": [5]},
    5: {"name": "MS", "description": "Mudstone", "adjacent": [4, 6]},
    6: {"name": "WS", "description": "Wackestone", "adjacent": [5, 7, 8]},
    7: {"name": "D", "description": "Dolomite", "adjacent": [6, 8]},
    8: {"name": "PS", "description": "Packstone-grainstone", "adjacent": [6, 7, 9]},
    9: {"name": "BS", "description": "Phylloid-algal bafflestone", "adjacent": [7, 8]},
}


@dataclass
class FaciesClassificationResult:
    """Container for facies classification results.

    Attributes:
        predictions: Predicted facies labels
        probabilities: Class probabilities (if available)
        accuracy: Overall accuracy
        f1_score: F1 score
        adjacent_accuracy: Accuracy counting adjacent facies as correct
        confusion_matrix: Confusion matrix
        classification_report: Detailed classification report
    """

    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    accuracy: float
    f1_score: float
    adjacent_accuracy: Optional[float]
    confusion_matrix: np.ndarray
    classification_report: str

    def __str__(self) -> str:
        return (
            f"Facies Classification Results:\n"
            f"  Accuracy: {self.accuracy:.3f}\n"
            f"  F1 Score: {self.f1_score:.3f}\n"
            f"  Adjacent Accuracy: {self.adjacent_accuracy:.3f if self.adjacent_accuracy else 'N/A'}\n"
            f"\n{self.classification_report}"
        )


class FaciesClassifier:
    """
    Facies classification using machine learning.

    Supports multiple algorithms: SVM, Random Forest, Gradient Boosting.
    Handles feature scaling, model training, and evaluation.
    """

    def __init__(
        self,
        algorithm: str = "svm",
        feature_names: Optional[list[str]] = None,
        random_state: int = 42,
    ):
        """
        Initialize facies classifier.

        Args:
            algorithm: 'svm', 'random_forest', or 'gradient_boosting'
            feature_names: List of feature column names
            random_state: Random seed for reproducibility
        """
        valid_algorithms = ["svm", "random_forest", "gradient_boosting"]
        if algorithm not in valid_algorithms:
            raise InvalidParameterError(
                f"Unknown algorithm: {algorithm}", valid_values=valid_algorithms
            )

        self.algorithm = algorithm
        self.feature_names = feature_names or [
            "GR",
            "ILD_log10",
            "DeltaPHI",
            "PHIND",
            "PE",
            "NM_M",
            "RELPOS",
        ]
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def _create_model(self, **kwargs):
        """Create model based on algorithm choice."""
        if self.algorithm == "svm":
            # Support Vector Machine with RBF kernel
            return SVC(
                C=kwargs.get("C", 10),
                gamma=kwargs.get("gamma", 1),
                probability=True,
                random_state=self.random_state,
            )
        elif self.algorithm == "random_forest":
            # Random Forest Classifier
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 20),
                min_samples_split=kwargs.get("min_samples_split", 5),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.algorithm == "gradient_boosting":
            # Gradient Boosting Classifier
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 5),
                random_state=self.random_state,
            )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **model_params,
    ):
        """
        Train facies classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Facies labels (n_samples,)
            **model_params: Model-specific parameters
        """
        # Validate inputs
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        if len(X) != len(y):
            raise DataValidationError(
                f"X ({len(X)}) and y ({len(y)}) must have same length",
                suggestion="Check data alignment",
            )

        if len(X) < 50:
            warnings.warn(
                f"Only {len(X)} training samples. Consider using more data.",
                UserWarning,
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model(**model_params)
        self.model.fit(X_scaled, y)

        self.is_fitted = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_proba: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Predict facies labels.

        Args:
            X: Feature matrix
            return_proba: Whether to return class probabilities

        Returns:
            predictions: Predicted facies labels
            probabilities: Class probabilities (if return_proba=True)
        """
        if not self.is_fitted:
            raise DataValidationError(
                "Model not fitted", suggestion="Call fit() before predict()"
            )

        # Extract features
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        if return_proba:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_scaled)
                return predictions, probabilities
            else:
                warnings.warn(
                    f"{self.algorithm} does not support probability estimates",
                    UserWarning,
                )
                return predictions, None

        return predictions

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        adjacent_facies: bool = True,
    ) -> FaciesClassificationResult:
        """
        Evaluate classifier on test data.

        Args:
            X_test: Test features
            y_test: True facies labels
            adjacent_facies: Count adjacent facies as correct

        Returns:
            FaciesClassificationResult with metrics
        """
        # Predict
        predictions, probabilities = self.predict(X_test, return_proba=True)

        # Compute metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        cm = confusion_matrix(y_test, predictions)

        # Classification report
        target_names = [FACIES_LABELS[i]["name"] for i in sorted(FACIES_LABELS.keys())]
        report = classification_report(
            y_test, predictions, target_names=target_names, zero_division=0
        )

        # Adjacent facies accuracy
        adj_accuracy = None
        if adjacent_facies:
            adj_accuracy = self._adjacent_accuracy(y_test, predictions)

        return FaciesClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            accuracy=accuracy,
            f1_score=f1,
            adjacent_accuracy=adj_accuracy,
            confusion_matrix=cm,
            classification_report=report,
        )

    def _adjacent_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate accuracy counting adjacent facies as correct.

        Adjacent facies are those that gradually blend into each other.
        """
        correct = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1
            elif true in FACIES_LABELS and pred in FACIES_LABELS[true]["adjacent"]:
                correct += 1

        return correct / len(y_true)

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
    ) -> dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Facies labels
            cv: Number of folds

        Returns:
            Dictionary with mean and std of scores
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        X_scaled = self.scaler.fit_transform(X)

        model = self._create_model()
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_weighted")

        return {"mean_f1": scores.mean(), "std_f1": scores.std(), "scores": scores}

    def hyperparameter_search(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[dict] = None,
        cv: int = 5,
    ) -> dict:
        """
        Grid search for optimal hyperparameters.

        Args:
            X: Feature matrix
            y: Facies labels
            param_grid: Parameter grid to search
            cv: Number of CV folds

        Returns:
            Dictionary with best parameters and score
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        X_scaled = self.scaler.fit_transform(X)

        # Default parameter grids
        if param_grid is None:
            if self.algorithm == "svm":
                param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}
            elif self.algorithm == "random_forest":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                }
            elif self.algorithm == "gradient_boosting":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }

        model = self._create_model()

        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_scaled, y)

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

    def cluster_unlabeled_data(
        self,
        X_unlabeled: Union[pd.DataFrame, np.ndarray],
        n_clusters: int = 9,
        method: str = "kmeans",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cluster unlabeled well data to find natural groupings.

        Useful for exploring wells without facies labels and identifying
        patterns that may correspond to lithologies.

        Args:
            X_unlabeled: Unlabeled feature data
            n_clusters: Number of clusters (typically matches number of facies)
            method: 'kmeans' or 'dbscan'

        Returns:
            cluster_labels: Cluster assignment for each sample
            cluster_centers: Cluster centroids (for kmeans)
        """
        if isinstance(X_unlabeled, pd.DataFrame):
            X_unlabeled = X_unlabeled[self.feature_names].values

        X_scaled = self.scaler.fit_transform(X_unlabeled)

        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = clusterer.cluster_centers_
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=10)
            cluster_labels = clusterer.fit_predict(X_scaled)
            # DBSCAN doesn't have explicit centers
            cluster_centers = None
        else:
            raise InvalidParameterError(
                f"Unknown clustering method: {method}",
                "Valid methods: 'kmeans', 'dbscan'",
            )

        return cluster_labels, cluster_centers

    def active_learning_query(
        self,
        X_unlabeled: Union[pd.DataFrame, np.ndarray],
        n_samples: int = 10,
        strategy: str = "uncertainty",
    ) -> np.ndarray:
        """
        Identify most informative unlabeled samples for expert labeling.

        Active learning helps prioritize which samples would most improve
        the model if labeled, maximizing value of limited expert time.

        Args:
            X_unlabeled: Unlabeled feature data
            n_samples: Number of samples to query
            strategy: 'uncertainty', 'margin', or 'entropy'

        Returns:
            indices: Indices of samples to label (ranked by informativeness)
        """
        if not self.is_fitted:
            raise DataValidationError(
                "Model not fitted", "Train model on labeled data before active learning"
            )

        if isinstance(X_unlabeled, pd.DataFrame):
            X_unlabeled = X_unlabeled[self.feature_names].values

        X_scaled = self.scaler.transform(X_unlabeled)

        if not hasattr(self.model, "predict_proba"):
            raise InvalidParameterError(
                f"{self.algorithm} does not support probability estimates",
                "Use 'svm' or 'random_forest' for active learning",
            )

        probas = self.model.predict_proba(X_scaled)

        if strategy == "uncertainty":
            # Select samples with lowest maximum probability
            max_probas = np.max(probas, axis=1)
            scores = 1 - max_probas
        elif strategy == "margin":
            # Select samples with smallest margin between top 2 classes
            sorted_probas = np.sort(probas, axis=1)
            margins = sorted_probas[:, -1] - sorted_probas[:, -2]
            scores = 1 - margins
        elif strategy == "entropy":
            # Select samples with highest entropy
            eps = 1e-10
            entropies = -np.sum(probas * np.log(probas + eps), axis=1)
            scores = entropies
        else:
            raise InvalidParameterError(
                f"Unknown active learning strategy: {strategy}",
                "Valid strategies: 'uncertainty', 'margin', 'entropy'",
            )

        # Return indices of top n_samples
        query_indices = np.argsort(scores)[::-1][:n_samples]
        return query_indices

    def semi_supervised_fit(
        self,
        X_labeled: Union[pd.DataFrame, np.ndarray],
        y_labeled: Union[pd.Series, np.ndarray],
        X_unlabeled: Union[pd.DataFrame, np.ndarray],
        alpha: float = 0.2,
    ):
        """
        Train using both labeled and unlabeled data.

        Uses label propagation to leverage structure in unlabeled data,
        improving performance when labeled data is limited.

        Args:
            X_labeled: Labeled feature data
            y_labeled: Labels for labeled data
            X_unlabeled: Unlabeled feature data
            alpha: Clamping factor (0-1). Higher = trust labels more
        """
        if isinstance(X_labeled, pd.DataFrame):
            X_labeled = X_labeled[self.feature_names].values
        if isinstance(X_unlabeled, pd.DataFrame):
            X_unlabeled = X_unlabeled[self.feature_names].values

        # Combine labeled and unlabeled data
        X_combined = np.vstack([X_labeled, X_unlabeled])

        # Create label array with -1 for unlabeled
        y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])

        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)

        # Train label spreading model
        label_spreader = LabelSpreading(
            kernel="knn",
            alpha=alpha,
            n_neighbors=7,
        )
        label_spreader.fit(X_scaled, y_combined)

        # Get pseudo-labels for unlabeled data
        y_pseudo = label_spreader.predict(X_scaled[len(X_labeled) :])

        # Get confidence scores
        label_distributions = label_spreader.label_distributions_[len(X_labeled) :]
        confidences = np.max(label_distributions, axis=1)

        # Filter high-confidence pseudo-labels
        high_conf_mask = confidences > 0.7

        # Combine original labels with high-confidence pseudo-labels
        X_augmented = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
        y_augmented = np.concatenate([y_labeled, y_pseudo[high_conf_mask]])

        # Train final model on augmented data
        X_aug_scaled = self.scaler.fit_transform(X_augmented)
        self.model = self._create_model()
        self.model.fit(X_aug_scaled, y_augmented)

        self.is_fitted = True

        n_pseudo = high_conf_mask.sum()
        print(
            f"Semi-supervised learning: Added {n_pseudo} high-confidence pseudo-labels"
        )

        return self

    def transfer_learning_fit(
        self,
        X_source: Union[pd.DataFrame, np.ndarray],
        y_source: Union[pd.Series, np.ndarray],
        X_target: Union[pd.DataFrame, np.ndarray],
        y_target: Union[pd.Series, np.ndarray],
        fine_tune_epochs: int = 50,
    ):
        """
        Transfer learning from source basin to target basin.

        Trains on source basin data first, then fine-tunes on limited
        target basin data. Useful when source basin has extensive labels
        but target basin has few.

        Args:
            X_source: Features from source basin (well-labeled)
            y_source: Labels from source basin
            X_target: Features from target basin (limited labels)
            y_target: Labels from target basin
            fine_tune_epochs: For gradient boosting, number of additional trees
        """
        if isinstance(X_source, pd.DataFrame):
            X_source = X_source[self.feature_names].values
        if isinstance(X_target, pd.DataFrame):
            X_target = X_target[self.feature_names].values

        # Phase 1: Pre-train on source data
        print("Phase 1: Training on source basin...")
        X_source_scaled = self.scaler.fit_transform(X_source)

        self.model = self._create_model()
        self.model.fit(X_source_scaled, y_source)

        # Phase 2: Fine-tune on target data
        print("Phase 2: Fine-tuning on target basin...")

        # Combine source and target, but weight target more heavily
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.concatenate([y_source, y_target])

        # Create sample weights: higher for target basin
        source_weight = 0.3
        target_weight = 1.0
        sample_weights = np.concatenate(
            [
                np.full(len(X_source), source_weight),
                np.full(len(X_target), target_weight),
            ]
        )

        X_combined_scaled = self.scaler.transform(X_combined)

        if self.algorithm in ["random_forest", "gradient_boosting"]:
            # Re-train with combined data and sample weights
            self.model.fit(X_combined_scaled, y_combined, sample_weight=sample_weights)
        else:
            # For SVM, just retrain on combined data
            self.model.fit(X_combined_scaled, y_combined)

        self.is_fitted = True

        print(
            f"Transfer learning complete: {len(X_source)} source + {len(X_target)} target samples"
        )

        return self


def load_facies_data(filepath: str) -> pd.DataFrame:
    """
    Load facies classification dataset.

    Expected columns: Facies, Depth, GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with facies data
    """
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        raise DataValidationError(
            f"Failed to load facies data: {str(e)}",
            suggestion="Check file path and format",
        )

    # Validate required columns
    required_cols = ["Facies", "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}",
            suggestion="Ensure dataset has standard facies classification columns",
        )

    return data


def prepare_facies_features(
    data: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
    test_well: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Prepare features for facies classification.

    Args:
        data: DataFrame with facies data
        feature_names: List of feature columns
        test_well: Well name to hold out for testing

    Returns:
        X_train, y_train, X_test, y_test
    """
    if feature_names is None:
        feature_names = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]

    # Remove test well if specified
    if test_well and "Well Name" in data.columns:
        test_data = data[data["Well Name"] == test_well]
        train_data = data[data["Well Name"] != test_well]

        X_train = train_data[feature_names]
        y_train = train_data["Facies"]
        X_test = test_data[feature_names]
        y_test = test_data["Facies"]

        return X_train, y_train, X_test, y_test
    else:
        X = data[feature_names]
        y = data["Facies"]
        return X, y, None, None
