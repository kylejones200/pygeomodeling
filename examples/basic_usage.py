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

from spe9_geomodeling import UnifiedSPE9Toolkit, load_spe9_data

# Configure logging
logger = logging.getLogger(__name__)


def main():
    """Run basic geomodeling example."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("SPE9 Geomodeling Toolkit - Basic Usage Example")

    # Load SPE9 data
    logger.info("Loading SPE9 dataset...")
    try:
        data = load_spe9_data()
        logger.info("Loaded SPE9 data: %s grid", data['grid_shape'])
        logger.info("   Properties: %s", list(data['properties'].keys()))
    except FileNotFoundError:
        logger.error("SPE9.GRDECL file not found. Please ensure the data file is available.")
        logger.info("   The bundled data file should be automatically detected.")
        return

    # Create toolkit
    logger.info("Setting up toolkit...")
    toolkit = UnifiedSPE9Toolkit()
    toolkit.load_spe9_data(data)

    # Create train/test split
    logger.info("Creating train/test split...")
    X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
        test_size=0.2, random_state=42
    )
    logger.info("   Training samples: %d", len(X_train))
    logger.info("   Test samples: %d", len(X_test))

    # Train a simple GP model
    logger.info("Training Gaussian Process model...")
    model = toolkit.create_sklearn_model("gpr", kernel_type="rbf")
    toolkit.train_sklearn_model(model, "rbf_gpr")

    # Evaluate the model
    logger.info("Evaluating model performance...")
    results = toolkit.evaluate_model("rbf_gpr", X_test, y_test)

    logger.info("   R² Score: %.4f", results.r2)
    logger.info("   RMSE: %.2f", results.rmse)
    logger.info("   MAE: %.2f", results.mae)

    # Make predictions on full grid
    logger.info("Making predictions on full grid...")
    predictions = toolkit.predict_full_grid("rbf_gpr")
    logger.info("   Predicted %d grid points", len(predictions))

    logger.info("Basic example completed successfully!")
    logger.info("Note: Try running the Deep GP experiment for advanced comparisons:")
    logger.info("   python -c 'from spe9_geomodeling import DeepGPExperiment; DeepGPExperiment().run_comparison_experiment()'")


if __name__ == "__main__":
    main()
