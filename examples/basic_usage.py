#!/usr/bin/env python3
"""
Basic usage example for SPE9 Geomodeling Toolkit.

This example demonstrates how to use the toolkit for basic geomodeling tasks.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from spe9_geomodeling import load_spe9_data, SPE9Toolkit


def main():
    """Run basic geomodeling example."""
    print("🚀 SPE9 Geomodeling Toolkit - Basic Usage Example")
    print("=" * 60)

    # Load SPE9 data
    print("📂 Loading SPE9 dataset...")
    try:
        data = load_spe9_data()
        print(f"✅ Loaded SPE9 data: {data['grid_shape']} grid")
        print(f"   Properties: {list(data['properties'].keys())}")
    except FileNotFoundError:
        print(
            "❌ SPE9.GRDECL file not found. Please ensure the data file is available."
        )
        print("   The bundled data file should be automatically detected.")
        return

    # Create toolkit
    print("\n🔧 Setting up toolkit...")
    toolkit = SPE9Toolkit()
    toolkit.load_data()
    toolkit.prepare_features()

    # Create train/test split
    print("📊 Creating train/test split...")
    X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
        test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Train a simple GP model
    print("\n🤖 Training Gaussian Process model...")
    model = toolkit.create_sklearn_model("gpr", kernel_type="rbf")
    toolkit.train_sklearn_model(model, "rbf_gpr")

    # Evaluate the model
    print("📈 Evaluating model performance...")
    results = toolkit.evaluate_model("rbf_gpr", X_test, y_test)

    print(f"   R² Score: {results.r2:.4f}")
    print(f"   RMSE: {results.rmse:.2f}")
    print(f"   MAE: {results.mae:.2f}")

    # Make predictions on full grid
    print("\n🔮 Making predictions on full grid...")
    predictions = toolkit.predict_full_grid("rbf_gpr")
    print(f"   Predicted {len(predictions)} grid points")

    print("\n✅ Basic example completed successfully!")
    print("💡 Try running the Deep GP experiment for advanced comparisons:")
    print(
        "   python -c 'from spe9_geomodeling import DeepGPExperiment; DeepGPExperiment().run_comparison_experiment()'"
    )


if __name__ == "__main__":
    main()
