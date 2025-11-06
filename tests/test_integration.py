"""Integration tests for the SPE9 geomodeling package."""

from unittest.mock import patch

import numpy as np
import pytest

from spe9_geomodeling import SPE9Plotter, SPE9Toolkit, UnifiedSPE9Toolkit


class TestPackageImports:
    """Test that all main components can be imported."""

    def test_main_imports(self):
        """Test importing main classes."""
        from spe9_geomodeling import SPE9Plotter, SPE9Toolkit, UnifiedSPE9Toolkit

        assert SPE9Toolkit is not None
        assert UnifiedSPE9Toolkit is not None
        assert SPE9Plotter is not None

    def test_parser_imports(self):
        """Test importing parser components."""
        from spe9_geomodeling import load_spe9_data

        assert load_spe9_data is not None

    def test_optional_imports(self):
        """Test optional imports don't break package."""
        try:
            from spe9_geomodeling import SPE9GPModel

            # If import succeeds, GPyTorch is available
            assert SPE9GPModel is not None
        except ImportError:
            # GPyTorch not available, should not break package
            pass

    def test_package_metadata(self):
        """Test package metadata is accessible."""
        import spe9_geomodeling

        assert hasattr(spe9_geomodeling, "__version__")
        assert hasattr(spe9_geomodeling, "__author__")
        assert hasattr(spe9_geomodeling, "__email__")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @patch("spe9_geomodeling.spe9_toolkit.load_spe9_data")
    @patch("pathlib.Path.exists")
    def test_spe9_toolkit_complete_workflow(
        self, mock_exists, mock_load, sample_grdecl_data
    ):
        """Test complete SPE9Toolkit workflow."""
        mock_load.return_value = sample_grdecl_data
        mock_exists.return_value = True

        # Initialize toolkit
        toolkit = SPE9Toolkit()

        # Complete workflow
        data = toolkit.load_data()
        assert data is not None

        grid_data = toolkit.prepare_features(add_geological_features=True)
        assert grid_data is not None

        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            test_size=0.2
        )
        assert len(X_train) > 0
        assert len(X_test) > 0

        x_scaler, y_scaler = toolkit.setup_scalers()
        assert x_scaler is not None
        assert y_scaler is not None

        # Train multiple models
        gpr = toolkit.create_model("gpr")
        rf = toolkit.create_model("rf")

        toolkit.train_model(gpr, "GPR")
        toolkit.train_model(rf, "RF")

        # Evaluate models
        gpr_results = toolkit.evaluate_model("GPR")
        rf_results = toolkit.evaluate_model("RF")

        assert gpr_results is not None
        assert rf_results is not None
        assert hasattr(gpr_results, "r2")
        assert hasattr(rf_results, "r2")

    @patch("spe9_geomodeling.unified_toolkit.load_spe9_data")
    @patch("pathlib.Path.exists")
    def test_unified_toolkit_sklearn_workflow(
        self, mock_exists, mock_load, sample_grdecl_data
    ):
        """Test complete UnifiedSPE9Toolkit sklearn workflow."""
        mock_load.return_value = sample_grdecl_data
        mock_exists.return_value = True

        # Initialize toolkit
        toolkit = UnifiedSPE9Toolkit(backend="sklearn")

        # Complete workflow
        data = toolkit.load_data()
        assert data is not None

        X, y = toolkit.prepare_features()
        assert X is not None
        assert y is not None

        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            test_size=0.2
        )
        assert len(X_train) > 0

        x_scaler, y_scaler = toolkit.setup_scalers()
        assert x_scaler is not None

        # Train and evaluate model
        gpr = toolkit.create_sklearn_model("gpr")
        toolkit.train_sklearn_model(gpr, "GPR")
        results = toolkit.evaluate_model("GPR")

        assert results is not None
        assert "r2" in results

    def test_plotting_integration(self, sample_grdecl_data):
        """Test plotting integration with toolkit results."""
        plotter = SPE9Plotter()

        # Create sample 3D data
        permx_3d = sample_grdecl_data["properties"]["PERMX"]

        # Test basic plotting
        fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=2)
        assert fig is not None
        assert ax is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestDataCompatibility:
    """Test compatibility with different data formats and sizes."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        small_data = {
            "dimensions": (3, 3, 2),
            "properties": {
                "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(3, 3, 2))
            },
        }

        with patch(
            "spe9_geomodeling.spe9_toolkit.load_spe9_data", return_value=small_data
        ):
            toolkit = SPE9Toolkit()
            toolkit.load_data()
            grid_data = toolkit.prepare_features()

            # Should work even with small dataset
            assert grid_data is not None
            assert len(grid_data.y_grid) == 18  # 3*3*2

    def test_large_dataset_simulation(self):
        """Test with simulated large dataset."""
        # Simulate larger dataset (but not too large for testing)
        large_data = {
            "dimensions": (20, 15, 10),
            "properties": {
                "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(20, 15, 10))
            },
        }

        with patch(
            "spe9_geomodeling.spe9_toolkit.load_spe9_data", return_value=large_data
        ):
            toolkit = SPE9Toolkit()
            toolkit.load_data()
            grid_data = toolkit.prepare_features()

            assert grid_data is not None
            assert len(grid_data.y_grid) == 3000  # 20*15*10

    def test_missing_properties(self):
        """Test handling of missing properties."""
        incomplete_data = {
            "dimensions": (5, 4, 3),
            "properties": {
                "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(5, 4, 3))
                # Missing other properties like PORO, PERMY, etc.
            },
        }

        with patch(
            "spe9_geomodeling.spe9_toolkit.load_spe9_data", return_value=incomplete_data
        ):
            toolkit = SPE9Toolkit()
            data = toolkit.load_data()

            # Should still work with just PERMX
            assert data is not None
            assert "PERMX" in data["properties"]


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        toolkit = SPE9Toolkit(data_path="nonexistent_file.grdecl")

        with pytest.raises(FileNotFoundError):
            toolkit.load_data()

    def test_workflow_order_enforcement(self):
        """Test that workflow steps must be done in order."""
        toolkit = SPE9Toolkit()

        # Should fail if trying to prepare features without loading data
        with pytest.raises(ValueError):
            toolkit.prepare_features()

        # Should fail if trying to create train/test split without features
        with pytest.raises(ValueError):
            toolkit.create_train_test_split()

        # Should fail if trying to setup scalers without train/test split
        with pytest.raises(ValueError):
            toolkit.setup_scalers()

    def test_model_not_found_error(self):
        """Test error when trying to evaluate non-existent model."""
        toolkit = SPE9Toolkit()

        with pytest.raises(ValueError, match="Model GPR not found"):
            toolkit.evaluate_model("GPR")

    def test_invalid_model_type(self):
        """Test error with invalid model type."""
        toolkit = SPE9Toolkit()

        with pytest.raises(ValueError, match="Unknown model type"):
            toolkit.create_model("invalid_model_type")


class TestPerformanceAndMemory:
    """Test performance and memory considerations."""

    @patch("spe9_geomodeling.spe9_toolkit.load_spe9_data")
    @patch("pathlib.Path.exists")
    def test_memory_efficient_processing(self, mock_exists, mock_load):
        """Test that processing doesn't consume excessive memory."""
        # Create moderately sized test data
        test_data = {
            "dimensions": (15, 12, 8),
            "properties": {
                "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(15, 12, 8))
            },
        }
        mock_load.return_value = test_data
        mock_exists.return_value = True

        toolkit = SPE9Toolkit()
        toolkit.load_data()

        # Should be able to process without memory errors
        grid_data = toolkit.prepare_features(add_geological_features=True)
        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            test_size=0.2
        )

        # Check that data sizes are reasonable
        assert len(X_train) < 2000  # Should be manageable size
        assert X_train.nbytes < 1e6  # Less than 1MB

    def test_model_training_time(self):
        """Test that model training completes in reasonable time."""
        import time

        # Create small dataset for quick training
        small_data = {
            "dimensions": (5, 4, 3),
            "properties": {
                "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(5, 4, 3))
            },
        }

        with patch(
            "spe9_geomodeling.spe9_toolkit.load_spe9_data", return_value=small_data
        ):
            toolkit = SPE9Toolkit()
            toolkit.load_data()
            toolkit.prepare_features()
            toolkit.create_train_test_split()
            toolkit.setup_scalers()

            # Time the model training
            start_time = time.time()
            gpr = toolkit.create_model("gpr")
            toolkit.train_model(gpr, "GPR")
            end_time = time.time()

            # Should complete quickly for small dataset
            training_time = end_time - start_time
            assert training_time < 10.0  # Less than 10 seconds


class TestCrossCompatibility:
    """Test compatibility between different components."""

    @patch("spe9_geomodeling.spe9_toolkit.load_spe9_data")
    @patch("spe9_geomodeling.unified_toolkit.load_spe9_data")
    def test_toolkit_interoperability(
        self, mock_unified_load, mock_spe9_load, sample_grdecl_data
    ):
        """Test that both toolkits work with same data."""
        mock_spe9_load.return_value = sample_grdecl_data
        mock_unified_load.return_value = sample_grdecl_data

        # Test SPE9Toolkit
        spe9_toolkit = SPE9Toolkit()
        spe9_data = spe9_toolkit.load_data()
        spe9_features = spe9_toolkit.prepare_features()

        # Test UnifiedSPE9Toolkit
        unified_toolkit = UnifiedSPE9Toolkit(backend="sklearn")
        unified_data = unified_toolkit.load_data()
        unified_X, unified_y = unified_toolkit.prepare_features()

        # Both should process the same data successfully
        assert spe9_data == unified_data
        assert spe9_features.X_grid.shape[0] == unified_X.shape[0]

    def test_plotter_with_toolkit_results(self, sample_grdecl_data):
        """Test that plotter works with toolkit results."""
        with patch(
            "spe9_geomodeling.spe9_toolkit.load_spe9_data",
            return_value=sample_grdecl_data,
        ):
            toolkit = SPE9Toolkit()
            toolkit.load_data()
            grid_data = toolkit.prepare_features()

            # Test that plotter can use toolkit data
            plotter = SPE9Plotter()
            permx_3d = grid_data.permx_3d

            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=2)
            assert fig is not None

            # Clean up
            import matplotlib.pyplot as plt

            plt.close(fig)
