"""Tests for SPE9Toolkit functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from spe9_geomodeling.spe9_toolkit import SPE9Toolkit, GridData, ModelResults


class TestSPE9Toolkit:
    """Test cases for SPE9Toolkit class."""

    def test_toolkit_initialization(self):
        """Test toolkit initialization."""
        toolkit = SPE9Toolkit()
        assert toolkit.data is None
        assert toolkit.grid_data is None
        assert toolkit.models == {}
        assert toolkit.scalers == {}
        assert toolkit.results == {}

    def test_toolkit_initialization_with_path(self, sample_grdecl_file):
        """Test toolkit initialization with custom data path."""
        toolkit = SPE9Toolkit(data_path=sample_grdecl_file)
        assert str(toolkit.data_path) == sample_grdecl_file

    @patch("spe9_geomodeling.spe9_toolkit.load_spe9_data")
    @patch("pathlib.Path.exists")
    def test_load_data(self, mock_exists, mock_load, sample_grdecl_data):
        """Test data loading."""
        mock_exists.return_value = True
        mock_load.return_value = sample_grdecl_data

        toolkit = SPE9Toolkit()
        result = toolkit.load_data()

        assert result == sample_grdecl_data
        assert toolkit.data == sample_grdecl_data
        mock_load.assert_called_once()

    def test_prepare_features_without_data(self):
        """Test prepare_features raises error without data."""
        toolkit = SPE9Toolkit()
        with pytest.raises(ValueError, match="Load data first"):
            toolkit.prepare_features()

    def test_prepare_features_basic(self, mock_spe9_toolkit):
        """Test basic feature preparation."""
        grid_data = mock_spe9_toolkit.prepare_features()

        assert isinstance(grid_data, GridData)
        assert grid_data.X_grid is not None
        assert grid_data.y_grid is not None
        assert grid_data.feature_names is not None
        assert len(grid_data.feature_names) >= 3  # At least x, y, z coordinates

    def test_prepare_features_with_geological(self, mock_spe9_toolkit):
        """Test feature preparation with geological features."""
        grid_data = mock_spe9_toolkit.prepare_features(add_geological_features=True)

        assert isinstance(grid_data, GridData)
        assert len(grid_data.feature_names) > 3  # Should have additional features
        expected_features = [
            "dist_center",
            "depth_factor",
            "xy_interaction",
            "xz_interaction",
            "yz_interaction",
        ]
        for feature in expected_features:
            assert feature in grid_data.feature_names

    def test_create_train_test_split_without_features(self):
        """Test train/test split raises error without features."""
        toolkit = SPE9Toolkit()
        with pytest.raises(ValueError, match="Prepare features first"):
            toolkit.create_train_test_split()

    def test_create_train_test_split(self, mock_spe9_toolkit):
        """Test train/test split creation."""
        mock_spe9_toolkit.prepare_features()
        X_train, X_test, y_train, y_test = mock_spe9_toolkit.create_train_test_split(
            test_size=0.3, random_state=42
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert X_train.shape[1] == X_test.shape[1]  # Same number of features

    def test_setup_scalers_without_split(self):
        """Test setup_scalers raises error without train/test split."""
        toolkit = SPE9Toolkit()
        with pytest.raises(ValueError, match="Create train/test split first"):
            toolkit.setup_scalers()

    def test_setup_scalers(self, mock_spe9_toolkit):
        """Test scaler setup."""
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split()

        x_scaler, y_scaler = mock_spe9_toolkit.setup_scalers()

        assert "x_scaler" in mock_spe9_toolkit.scalers
        assert "y_scaler" in mock_spe9_toolkit.scalers
        assert mock_spe9_toolkit.grid_data.X_train_scaled is not None
        assert mock_spe9_toolkit.grid_data.y_train_scaled is not None

    def test_create_model_gpr(self, mock_spe9_toolkit):
        """Test GPR model creation."""
        model = mock_spe9_toolkit.create_model("gpr")
        assert isinstance(model, GaussianProcessRegressor)

    def test_create_model_rf(self, mock_spe9_toolkit):
        """Test Random Forest model creation."""
        model = mock_spe9_toolkit.create_model("rf")
        assert isinstance(model, RandomForestRegressor)

    def test_create_model_invalid(self, mock_spe9_toolkit):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            mock_spe9_toolkit.create_model("invalid_model")

    def test_train_model_without_scalers(self, mock_spe9_toolkit):
        """Test train_model raises error without scalers."""
        model = mock_spe9_toolkit.create_model("gpr")
        with pytest.raises(ValueError, match="Setup scalers first"):
            mock_spe9_toolkit.train_model(model, "GPR")

    def test_train_model(self, mock_spe9_toolkit):
        """Test model training."""
        # Setup full pipeline
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split()
        mock_spe9_toolkit.setup_scalers()

        model = mock_spe9_toolkit.create_model("gpr")
        trained_model = mock_spe9_toolkit.train_model(model, "GPR")

        assert "GPR" in mock_spe9_toolkit.models
        assert mock_spe9_toolkit.models["GPR"] == trained_model

    def test_evaluate_model_not_found(self, mock_spe9_toolkit):
        """Test evaluate_model raises error for non-existent model."""
        with pytest.raises(ValueError, match="Model GPR not found"):
            mock_spe9_toolkit.evaluate_model("GPR")

    def test_evaluate_model(self, mock_spe9_toolkit):
        """Test model evaluation."""
        # Setup full pipeline
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split()
        mock_spe9_toolkit.setup_scalers()

        model = mock_spe9_toolkit.create_model("gpr")
        mock_spe9_toolkit.train_model(model, "GPR")

        results = mock_spe9_toolkit.evaluate_model("GPR")

        assert isinstance(results, ModelResults)
        assert hasattr(results, "r2")
        assert hasattr(results, "rmse")
        assert hasattr(results, "mae")
        assert hasattr(results, "y_pred")
        assert "GPR" in mock_spe9_toolkit.results


class TestGridData:
    """Test cases for GridData dataclass."""

    def test_grid_data_creation(self):
        """Test GridData creation."""
        X_grid = np.random.randn(100, 3)
        y_grid = np.random.randn(100)
        feature_names = ["x", "y", "z"]
        permx_3d = np.random.randn(5, 4, 5)
        dimensions = (5, 4, 5)

        grid_data = GridData(
            X_grid=X_grid,
            y_grid=y_grid,
            feature_names=feature_names,
            permx_3d=permx_3d,
            dimensions=dimensions,
        )

        assert np.array_equal(grid_data.X_grid, X_grid)
        assert np.array_equal(grid_data.y_grid, y_grid)
        assert grid_data.feature_names == feature_names
        assert np.array_equal(grid_data.permx_3d, permx_3d)
        assert grid_data.dimensions == dimensions


class TestModelResults:
    """Test cases for ModelResults dataclass."""

    def test_model_results_creation(self):
        """Test ModelResults creation."""
        r2 = 0.85
        rmse = 10.5
        mae = 8.2
        y_pred = np.random.randn(50)
        y_std = np.random.randn(50)

        results = ModelResults(r2=r2, rmse=rmse, mae=mae, y_pred=y_pred, y_std=y_std)

        assert results.r2 == r2
        assert results.rmse == rmse
        assert results.mae == mae
        assert np.array_equal(results.y_pred, y_pred)
        assert np.array_equal(results.y_std, y_std)

    def test_model_results_without_std(self):
        """Test ModelResults creation without standard deviation."""
        results = ModelResults(r2=0.85, rmse=10.5, mae=8.2, y_pred=np.random.randn(50))

        assert results.y_std is None


class TestSPE9ToolkitIntegration:
    """Integration tests for SPE9Toolkit."""

    def test_full_workflow_gpr(self, mock_spe9_toolkit):
        """Test complete workflow with GPR."""
        # Complete workflow
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split(test_size=0.2)
        mock_spe9_toolkit.setup_scalers()

        gpr = mock_spe9_toolkit.create_model("gpr")
        mock_spe9_toolkit.train_model(gpr, "GPR")
        results = mock_spe9_toolkit.evaluate_model("GPR")

        # Check that everything worked
        assert isinstance(results, ModelResults)
        assert results.r2 is not None
        assert results.rmse is not None
        assert results.mae is not None
        assert len(results.y_pred) > 0

    def test_full_workflow_rf(self, mock_spe9_toolkit):
        """Test complete workflow with Random Forest."""
        # Complete workflow
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split(test_size=0.2)
        mock_spe9_toolkit.setup_scalers()

        rf = mock_spe9_toolkit.create_model("rf")
        mock_spe9_toolkit.train_model(rf, "RF")
        results = mock_spe9_toolkit.evaluate_model("RF")

        # Check that everything worked
        assert isinstance(results, ModelResults)
        assert results.r2 is not None
        assert results.rmse is not None
        assert results.mae is not None
        assert len(results.y_pred) > 0

    def test_multiple_models(self, mock_spe9_toolkit):
        """Test training and evaluating multiple models."""
        # Setup
        mock_spe9_toolkit.prepare_features()
        mock_spe9_toolkit.create_train_test_split(test_size=0.2)
        mock_spe9_toolkit.setup_scalers()

        # Train multiple models
        gpr = mock_spe9_toolkit.create_model("gpr")
        rf = mock_spe9_toolkit.create_model("rf")

        mock_spe9_toolkit.train_model(gpr, "GPR")
        mock_spe9_toolkit.train_model(rf, "RF")

        # Evaluate both
        gpr_results = mock_spe9_toolkit.evaluate_model("GPR")
        rf_results = mock_spe9_toolkit.evaluate_model("RF")

        assert len(mock_spe9_toolkit.models) == 2
        assert len(mock_spe9_toolkit.results) == 2
        assert isinstance(gpr_results, ModelResults)
        assert isinstance(rf_results, ModelResults)
