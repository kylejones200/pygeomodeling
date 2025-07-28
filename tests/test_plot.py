"""Tests for plotting functionality."""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from spe9_geomodeling.plot import SPE9Plotter


class TestSPE9Plotter:
    """Test cases for SPE9Plotter class."""

    def test_plotter_initialization(self):
        """Test plotter initialization."""
        plotter = SPE9Plotter()
        assert plotter.figsize == (12, 8)
        assert plotter.dpi == 150
        assert plotter.colors is not None

    def test_plotter_initialization_custom(self):
        """Test plotter initialization with custom parameters."""
        plotter = SPE9Plotter(figsize=(10, 6), dpi=100)
        assert plotter.figsize == (10, 6)
        assert plotter.dpi == 100

    def test_plot_permeability_slice(self):
        """Test permeability slice plotting."""
        plotter = SPE9Plotter()

        # Create sample 3D permeability data
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        # Test plotting - should not raise errors
        try:
            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=2)
            assert fig is not None
            assert ax is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_permeability_slice raised an exception: {e}")

    def test_plot_permeability_slice_invalid_slice(self):
        """Test permeability slice plotting with invalid slice."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        # Test with slice index out of bounds
        with pytest.raises((IndexError, ValueError)):
            plotter.plot_permeability_slice(permx_3d, z_slice=10)

    def test_plot_prediction_comparison(self):
        """Test prediction comparison plotting."""
        plotter = SPE9Plotter()

        # Create sample data
        true_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))
        pred_3d = true_3d + np.random.normal(0, 0.1, size=(10, 8, 5))

        try:
            fig, axes = plotter.plot_prediction_comparison(true_3d, pred_3d, z_slice=2)
            assert fig is not None
            assert axes is not None
            assert len(axes) >= 2  # Should have at least true and predicted plots
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_prediction_comparison raised an exception: {e}")

    def test_plot_prediction_comparison_with_uncertainty(self):
        """Test prediction comparison plotting with uncertainty."""
        plotter = SPE9Plotter()

        # Create sample data
        true_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))
        pred_3d = true_3d + np.random.normal(0, 0.1, size=(10, 8, 5))
        sigma_3d = np.random.uniform(0.1, 0.5, size=(10, 8, 5))

        try:
            fig, axes = plotter.plot_prediction_comparison(
                true_3d, pred_3d, z_slice=2, sigma_3d=sigma_3d
            )
            assert fig is not None
            assert axes is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(
                f"plot_prediction_comparison with uncertainty raised an exception: {e}"
            )

    def test_plot_model_comparison(self):
        """Test model comparison plotting."""
        plotter = SPE9Plotter()

        # Create sample results
        results = {
            "GPR": {"r2": 0.85, "rmse": 10.5, "mae": 8.2},
            "RF": {"r2": 0.78, "rmse": 12.1, "mae": 9.5},
            "SVR": {"r2": 0.72, "rmse": 14.2, "mae": 11.1},
        }

        try:
            fig, axes = plotter.plot_model_comparison(results)
            assert fig is not None
            assert axes is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_model_comparison raised an exception: {e}")

    def test_plot_model_comparison_empty_results(self):
        """Test model comparison plotting with empty results."""
        plotter = SPE9Plotter()

        with pytest.raises((ValueError, KeyError)):
            plotter.plot_model_comparison({})

    def test_plot_training_curve(self):
        """Test training curve plotting."""
        plotter = SPE9Plotter()

        # Create sample training data
        iterations = np.arange(1, 101)
        loss_values = np.exp(-iterations / 20) + np.random.normal(0, 0.01, 100)

        try:
            fig, ax = plotter.plot_training_curve(iterations, loss_values)
            assert fig is not None
            assert ax is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_training_curve raised an exception: {e}")

    def test_plot_residuals(self):
        """Test residuals plotting."""
        plotter = SPE9Plotter()

        # Create sample data
        y_true = np.random.lognormal(mean=2.0, sigma=1.0, size=100)
        y_pred = y_true + np.random.normal(0, 0.1, size=100)

        try:
            fig, axes = plotter.plot_residuals(y_true, y_pred)
            assert fig is not None
            assert axes is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_residuals raised an exception: {e}")

    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        plotter = SPE9Plotter()

        # Create sample feature importance data
        feature_names = ["x_norm", "y_norm", "z_norm", "dist_center", "depth_factor"]
        importance_values = np.random.uniform(0.1, 1.0, len(feature_names))

        try:
            fig, ax = plotter.plot_feature_importance(feature_names, importance_values)
            assert fig is not None
            assert ax is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised an exception: {e}")

    def test_plot_feature_importance_mismatched_lengths(self):
        """Test feature importance plotting with mismatched lengths."""
        plotter = SPE9Plotter()

        feature_names = ["x_norm", "y_norm", "z_norm"]
        importance_values = np.array([0.5, 0.3])  # Different length

        with pytest.raises((ValueError, IndexError)):
            plotter.plot_feature_importance(feature_names, importance_values)


class TestSPE9PlotterCustomization:
    """Test plotting customization options."""

    def test_custom_colormap(self):
        """Test custom colormap usage."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        try:
            fig, ax = plotter.plot_permeability_slice(
                permx_3d, z_slice=2, cmap="viridis"
            )
            assert fig is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Custom colormap plotting raised an exception: {e}")

    def test_custom_title(self):
        """Test custom title setting."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        try:
            fig, ax = plotter.plot_permeability_slice(
                permx_3d, z_slice=2, title="Custom Title"
            )
            assert fig is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Custom title plotting raised an exception: {e}")

    def test_save_figure(self, tmp_path):
        """Test figure saving functionality."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        output_file = tmp_path / "test_plot.png"

        try:
            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=2)
            fig.savefig(output_file, dpi=plotter.dpi, bbox_inches="tight")
            plt.close(fig)

            # Check that file was created
            assert output_file.exists()
            assert output_file.stat().st_size > 0
        except Exception as e:
            pytest.fail(f"Figure saving raised an exception: {e}")


class TestSPE9PlotterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test plotting with empty data."""
        plotter = SPE9Plotter()

        with pytest.raises((ValueError, IndexError)):
            plotter.plot_permeability_slice(np.array([]), z_slice=0)

    def test_1d_data(self):
        """Test plotting with 1D data instead of 3D."""
        plotter = SPE9Plotter()
        data_1d = np.random.randn(100)

        with pytest.raises((ValueError, IndexError)):
            plotter.plot_permeability_slice(data_1d, z_slice=0)

    def test_2d_data(self):
        """Test plotting with 2D data instead of 3D."""
        plotter = SPE9Plotter()
        data_2d = np.random.randn(10, 8)

        with pytest.raises((ValueError, IndexError)):
            plotter.plot_permeability_slice(data_2d, z_slice=0)

    def test_negative_slice_index(self):
        """Test plotting with negative slice index."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))

        # Negative indices should work (Python indexing)
        try:
            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=-1)
            assert fig is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Negative slice index raised an exception: {e}")

    def test_nan_values(self):
        """Test plotting with NaN values."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))
        permx_3d[0, 0, 0] = np.nan  # Add some NaN values

        try:
            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=0)
            assert fig is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"NaN values in data raised an exception: {e}")

    def test_infinite_values(self):
        """Test plotting with infinite values."""
        plotter = SPE9Plotter()
        permx_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))
        permx_3d[0, 0, 0] = np.inf  # Add some infinite values

        try:
            fig, ax = plotter.plot_permeability_slice(permx_3d, z_slice=0)
            assert fig is not None
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Infinite values in data raised an exception: {e}")


class TestSPE9PlotterIntegration:
    """Integration tests for plotting functionality."""

    def test_complete_visualization_workflow(self, tmp_path):
        """Test complete visualization workflow."""
        plotter = SPE9Plotter()

        # Create sample data for complete workflow
        true_3d = np.random.lognormal(mean=2.0, sigma=1.0, size=(10, 8, 5))
        pred_3d = true_3d + np.random.normal(0, 0.1, size=(10, 8, 5))
        sigma_3d = np.random.uniform(0.1, 0.5, size=(10, 8, 5))

        y_true = np.random.lognormal(mean=2.0, sigma=1.0, size=100)
        y_pred = y_true + np.random.normal(0, 0.1, size=100)

        results = {
            "GPR": {"r2": 0.85, "rmse": 10.5, "mae": 8.2},
            "RF": {"r2": 0.78, "rmse": 12.1, "mae": 9.5},
        }

        try:
            # Test multiple plot types
            fig1, ax1 = plotter.plot_permeability_slice(true_3d, z_slice=2)
            fig2, axes2 = plotter.plot_prediction_comparison(
                true_3d, pred_3d, z_slice=2, sigma_3d=sigma_3d
            )
            fig3, axes3 = plotter.plot_residuals(y_true, y_pred)
            fig4, axes4 = plotter.plot_model_comparison(results)

            # Save all figures
            for i, fig in enumerate([fig1, fig2, fig3, fig4], 1):
                output_file = tmp_path / f"test_plot_{i}.png"
                fig.savefig(output_file, dpi=plotter.dpi, bbox_inches="tight")
                plt.close(fig)
                assert output_file.exists()

        except Exception as e:
            pytest.fail(f"Complete visualization workflow raised an exception: {e}")
