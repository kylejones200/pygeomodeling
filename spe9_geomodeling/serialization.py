"""
Model serialization and versioning utilities.

Provides functionality to save/load trained models with metadata and versioning.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import joblib

from .exceptions import SerializationError, raise_invalid_parameter

# Configure logging
logger = logging.getLogger(__name__)


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelMetadata:
    """Metadata for serialized models."""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        backend: str,
        version: str = "1.0",
        **kwargs,
    ):
        """Initialize model metadata.

        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'gpr', 'random_forest', 'deep_gp')
            backend: Backend used ('sklearn' or 'gpytorch')
            version: Model version string
            **kwargs: Additional metadata fields
        """
        self.model_name = model_name
        self.model_type = model_type
        self.backend = backend
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        # Store additional metadata
        self.metadata = kwargs

        # Training information
        self.training_info = {}
        self.hyperparameters = {}
        self.performance_metrics = {}

    def add_training_info(
        self,
        n_samples: int,
        n_features: int,
        feature_names: list = None,
        training_time: float = None,
    ):
        """Add training information to metadata."""
        self.training_info = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names or [],
            "training_time_seconds": training_time,
        }
        self.updated_at = datetime.now().isoformat()

    def add_hyperparameters(self, hyperparameters: dict[str, Any]):
        """Add hyperparameters to metadata."""
        self.hyperparameters = hyperparameters
        self.updated_at = datetime.now().isoformat()

    def add_performance_metrics(self, metrics: dict[str, float]):
        """Add performance metrics to metadata."""
        self.performance_metrics = metrics
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "backend": self.backend,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "training_info": self.training_info,
            "hyperparameters": self.hyperparameters,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        metadata = cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            backend=data["backend"],
            version=data.get("version", "1.0"),
            **data.get("metadata", {}),
        )
        metadata.created_at = data.get("created_at", metadata.created_at)
        metadata.updated_at = data.get("updated_at", metadata.updated_at)
        metadata.training_info = data.get("training_info", {})
        metadata.hyperparameters = data.get("hyperparameters", {})
        metadata.performance_metrics = data.get("performance_metrics", {})
        return metadata

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(name={self.model_name}, type={self.model_type}, "
            f"backend={self.backend}, version={self.version})"
        )


class ModelSerializer:
    """Handles model serialization and deserialization."""

    SUPPORTED_FORMATS = ["joblib", "pickle", "torch"]

    def __init__(self, save_dir: Union[str, Path] = "saved_models"):
        """Initialize model serializer.

        Args:
            save_dir: Directory to save models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        scaler: Any = None,
        format: str = "joblib",
    ) -> Path:
        """Save a trained model with metadata.

        Args:
            model: Trained model object
            metadata: Model metadata
            scaler: Optional data scaler
            format: Serialization format ('joblib', 'pickle', or 'torch')

        Returns:
            Path to saved model directory

        Raises:
            SerializationError: If serialization fails
        """
        if format not in self.SUPPORTED_FORMATS:
            raise_invalid_parameter("format", format, self.SUPPORTED_FORMATS)

        # Create model directory
        model_dir = self.save_dir / metadata.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            if format == "torch":
                if not TORCH_AVAILABLE:
                    raise SerializationError(
                        "PyTorch is not available",
                        suggestion="Install PyTorch: pip install torch",
                    )
                model_path = model_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
            elif format == "joblib":
                model_path = model_dir / "model.joblib"
                joblib.dump(model, model_path)
            else:  # pickle
                model_path = model_dir / "model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # Save scaler if provided
            if scaler is not None:
                scaler_path = model_dir / "scaler.joblib"
                joblib.dump(scaler, scaler_path)

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Save version info
            version_path = model_dir / "VERSION"
            with open(version_path, "w") as f:
                f.write(metadata.version)

            logger.info("Model saved to: %s", model_dir)
            logger.info("  - Model file: %s", model_path.name)
            logger.info("  - Format: %s", format)
            if scaler is not None:
                logger.info("  - Scaler: saved")
            logger.info("  - Metadata: saved")

            return model_dir

        except Exception as e:
            raise SerializationError(
                f"Failed to save model: {str(e)}",
                suggestion="Check that you have write permissions and sufficient disk space",
            )

    def load_model(
        self, model_name: str, format: str = "joblib"
    ) -> tuple[Any, ModelMetadata, Optional[Any]]:
        """Load a saved model with metadata.

        Args:
            model_name: Name of the model to load
            format: Serialization format used

        Returns:
            Tuple of (model, metadata, scaler)

        Raises:
            SerializationError: If loading fails
        """
        model_dir = self.save_dir / model_name

        if not model_dir.exists():
            raise SerializationError(
                f"Model directory not found: {model_dir}",
                suggestion=f"Available models: {self.list_models()}",
            )

        try:
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise SerializationError(
                    f"Metadata file not found: {metadata_path}",
                    suggestion="The model directory may be corrupted",
                )

            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)

            # Load model
            if format == "torch":
                if not TORCH_AVAILABLE:
                    raise SerializationError(
                        "PyTorch is not available",
                        suggestion="Install PyTorch: pip install torch",
                    )
                model_path = model_dir / "model.pt"
                # Note: For torch models, we need the model architecture
                # This is a placeholder - actual implementation needs model class
                model = torch.load(model_path)
            elif format == "joblib":
                model_path = model_dir / "model.joblib"
                model = joblib.load(model_path)
            else:  # pickle
                model_path = model_dir / "model.pkl"
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

            # Load scaler if exists
            scaler_path = model_dir / "scaler.joblib"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            logger.info("Model loaded from: %s", model_dir)
            logger.info("  - Model: %s", metadata.model_name)
            logger.info("  - Type: %s", metadata.model_type)
            logger.info("  - Backend: %s", metadata.backend)
            logger.info("  - Version: %s", metadata.version)
            if metadata.performance_metrics:
                logger.info("  - Metrics: %s", metadata.performance_metrics)

            return model, metadata, scaler

        except Exception as e:
            raise SerializationError(
                f"Failed to load model: {str(e)}",
                suggestion="Check that the model files are not corrupted",
            )

    def list_models(self) -> list[str]:
        """List all saved models.

        Returns:
            List of model names
        """
        if not self.save_dir.exists():
            return []

        models = []
        for item in self.save_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                models.append(item.name)

        return sorted(models)

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a saved model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        model_dir = self.save_dir / model_name
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            raise SerializationError(
                f"Model not found: {model_name}",
                suggestion=f"Available models: {self.list_models()}",
            )

        with open(metadata_path) as f:
            return json.load(f)

    def delete_model(self, model_name: str) -> None:
        """Delete a saved model.

        Args:
            model_name: Name of the model to delete
        """
        model_dir = self.save_dir / model_name

        if not model_dir.exists():
            raise SerializationError(
                f"Model not found: {model_name}",
                suggestion=f"Available models: {self.list_models()}",
            )

        import shutil

        shutil.rmtree(model_dir)
        logger.info("Model deleted: %s", model_name)


# Convenience functions
def save_model(
    model: Any,
    model_name: str,
    model_type: str,
    backend: str = "sklearn",
    save_dir: Union[str, Path] = "saved_models",
    scaler: Any = None,
    metrics: dict[str, float] = None,
    **kwargs,
) -> Path:
    """Convenience function to save a model.

    Args:
        model: Trained model
        model_name: Name for the model
        model_type: Type of model
        backend: Backend used
        save_dir: Directory to save model
        scaler: Optional scaler
        metrics: Optional performance metrics
        **kwargs: Additional metadata

    Returns:
        Path to saved model directory
    """
    serializer = ModelSerializer(save_dir)
    metadata = ModelMetadata(model_name, model_type, backend, **kwargs)

    if metrics:
        metadata.add_performance_metrics(metrics)

    format = "torch" if backend == "gpytorch" else "joblib"
    return serializer.save_model(model, metadata, scaler, format)


def load_model(
    model_name: str, save_dir: Union[str, Path] = "saved_models"
) -> tuple[Any, ModelMetadata, Optional[Any]]:
    """Convenience function to load a model.

    Args:
        model_name: Name of the model
        save_dir: Directory containing saved models

    Returns:
        Tuple of (model, metadata, scaler)
    """
    serializer = ModelSerializer(save_dir)

    # Try to detect format from files
    model_dir = Path(save_dir) / model_name
    if (model_dir / "model.pt").exists():
        format = "torch"
    elif (model_dir / "model.joblib").exists():
        format = "joblib"
    else:
        format = "pickle"

    return serializer.load_model(model_name, format)
