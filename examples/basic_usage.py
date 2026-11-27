#!/usr/bin/env python3
"""
Basic usage example for SPE9 Geomodeling Toolkit.

This example demonstrates how to use the toolkit for basic geomodeling tasks.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygeomodeling import UnifiedSPE9Toolkit, load_spe9_data

# Configure logging
logger = logging.getLogger(__name__)


def main():
    """Run basic geomodeling example."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("SPE9 Geomodeling Toolkit - Basic Usage Example")

    # Load SPE9 data. The parser returns dimensions, not grid_shape.
    # The bundled GRDECL contains PERMX only.
    logger.info("Loading SPE9 dataset...")
    try:
        data = load_spe9_data()
        logger.info("Loaded SPE9 data: %s grid", data["dimensions"])
        logger.info("   Properties: %s", list(data["properties"].keys()))
    except FileNotFoundError:
        logger.error("SPE9.GRDECL file not found. Please ensure the data file is available.")
        logger.info("   The bundled data file should be automatically detected.")
        return

    # Create toolkit
    logger.info("Setting up toolkit...")
    toolkit = UnifiedSPE9Toolkit()
    toolkit.load_spe9_data(data)
    toolkit.prepare_features()

    # Subsample before the split. A Gaussian process on the full SPE9 grid
    # (about 9,000 cells) is too expensive for a first example. PERMX is
    # modeled as log1p(permeability); the bundled GRDECL has no porosity.
    logger.info("Creating train/test split...")
    X_train, X_test, _, _ = toolkit.create_train_test_split(
        test_size=0.2, train_size=400, random_state=42, log_transform=True
    )
    logger.info("   Training samples: %d", len(X_train))
    logger.info("   Test samples: %d", len(X_test))

    logger.info("Fitting scalers...")
    toolkit.setup_scalers()

    # Train a simple GP model
    logger.info("Training Gaussian Process model...")
    model = toolkit.create_sklearn_model(
        "gpr", kernel_type="rbf", n_restarts_optimizer=0, alpha=0.1
    )
    toolkit.train_sklearn_model(model, "rbf_gpr")

    # Evaluate the model. evaluate_model uses the split stored on the toolkit
    # and returns a dict.
    logger.info("Evaluating model performance...")
    results = toolkit.evaluate_model("rbf_gpr")

    logger.info("   R² Score (log1p PERMX): %.4f", results["r2"])
    logger.info("   RMSE: %.2f", results["rmse"])
    logger.info("   MAE: %.2f", results["mae"])

    # Make predictions on full grid
    logger.info("Making predictions on full grid...")
    predictions = toolkit.predict_full_grid("rbf_gpr")
    logger.info("   Predicted %d grid points", len(predictions))

    logger.info("Basic example completed successfully!")
    logger.info("Note: Try running the Deep GP experiment for advanced comparisons:")
    logger.info("   python -c 'from pygeomodeling import DeepGPExperiment; DeepGPExperiment().run_comparison_experiment()'")


if __name__ == "__main__":
    main()
