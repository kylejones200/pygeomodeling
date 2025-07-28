"""Tests for UnifiedSPE9Toolkit functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from spe9_geomodeling.unified_toolkit import UnifiedSPE9Toolkit


class TestUnifiedSPE9Toolkit:
    """Test cases for UnifiedSPE9Toolkit class."""
    
    def test_toolkit_initialization_sklearn(self):
        """Test toolkit initialization with sklearn backend."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        assert toolkit.backend == 'sklearn'
        assert toolkit.data is None
        assert toolkit.models == {}
        assert toolkit.scalers == {}
        assert toolkit.results == {}
    
    def test_toolkit_initialization_gpytorch(self, skip_if_no_gpytorch):
        """Test toolkit initialization with gpytorch backend."""
        toolkit = UnifiedSPE9Toolkit(backend='gpytorch')
        assert toolkit.backend == 'gpytorch'
    
    def test_toolkit_initialization_gpytorch_unavailable(self):
        """Test toolkit initialization with gpytorch when unavailable."""
        with patch('spe9_geomodeling.unified_toolkit.GPYTORCH_AVAILABLE', False):
            with pytest.raises(ValueError, match="GPyTorch backend requested but GPyTorch is not installed"):
                UnifiedSPE9Toolkit(backend='gpytorch')
    
    def test_toolkit_initialization_invalid_backend(self):
        """Test toolkit initialization with invalid backend."""
        with pytest.raises(ValueError, match="Backend must be 'sklearn' or 'gpytorch'"):
            UnifiedSPE9Toolkit(backend='invalid')
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_load_data(self, mock_load, sample_grdecl_data):
        """Test data loading."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        result = toolkit.load_data()
        
        assert result == sample_grdecl_data
        assert toolkit.data == sample_grdecl_data
        mock_load.assert_called_once()
    
    def test_prepare_features_without_data(self):
        """Test prepare_features raises error without data."""
        toolkit = UnifiedSPE9Toolkit()
        with pytest.raises(ValueError, match="Load data first"):
            toolkit.prepare_features()
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_prepare_features(self, mock_load, sample_grdecl_data):
        """Test feature preparation."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        toolkit.load_data()
        X, y = toolkit.prepare_features()
        
        assert X is not None
        assert y is not None
        assert X.shape[0] == y.shape[0]
        assert toolkit.X_grid is not None
        assert toolkit.y_grid is not None
    
    def test_create_train_test_split_without_features(self):
        """Test train/test split raises error without features."""
        toolkit = UnifiedSPE9Toolkit()
        with pytest.raises(ValueError, match="Prepare features first"):
            toolkit.create_train_test_split()
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_create_train_test_split(self, mock_load, sample_grdecl_data):
        """Test train/test split creation."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        toolkit.load_data()
        toolkit.prepare_features()
        
        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            test_size=0.3, random_state=42
        )
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_setup_scalers_without_split(self):
        """Test setup_scalers raises error without train/test split."""
        toolkit = UnifiedSPE9Toolkit()
        with pytest.raises(ValueError, match="Create train/test split first"):
            toolkit.setup_scalers()
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_setup_scalers(self, mock_load, sample_grdecl_data):
        """Test scaler setup."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        toolkit.load_data()
        toolkit.prepare_features()
        toolkit.create_train_test_split()
        
        x_scaler, y_scaler = toolkit.setup_scalers()
        
        assert "x_scaler" in toolkit.scalers
        assert "y_scaler" in toolkit.scalers
        assert toolkit.X_train_scaled is not None
        assert toolkit.y_train_scaled is not None


class TestUnifiedSPE9ToolkitSklearn:
    """Test sklearn-specific functionality."""
    
    def test_create_sklearn_model_gpr(self):
        """Test sklearn GPR model creation."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        model = toolkit.create_sklearn_model('gpr')
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        assert isinstance(model, GaussianProcessRegressor)
    
    def test_create_sklearn_model_rf(self):
        """Test sklearn Random Forest model creation."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        model = toolkit.create_sklearn_model('rf')
        
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)
    
    def test_create_sklearn_model_svr(self):
        """Test sklearn SVR model creation."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        model = toolkit.create_sklearn_model('svr')
        
        from sklearn.svm import SVR
        assert isinstance(model, SVR)
    
    def test_create_sklearn_model_invalid(self):
        """Test invalid sklearn model type."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        with pytest.raises(ValueError, match="Unknown sklearn model type"):
            toolkit.create_sklearn_model('invalid')
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_train_sklearn_model(self, mock_load, sample_grdecl_data):
        """Test sklearn model training."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        toolkit.load_data()
        toolkit.prepare_features()
        toolkit.create_train_test_split()
        toolkit.setup_scalers()
        
        model = toolkit.create_sklearn_model('gpr')
        trained_model = toolkit.train_sklearn_model(model, 'GPR')
        
        assert 'GPR' in toolkit.models
        assert toolkit.models['GPR'] == trained_model


class TestUnifiedSPE9ToolkitGPyTorch:
    """Test GPyTorch-specific functionality."""
    
    def test_create_gpytorch_model_requires_backend(self):
        """Test GPyTorch model creation requires gpytorch backend."""
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        with pytest.raises(ValueError, match="GPyTorch models require 'gpytorch' backend"):
            toolkit.create_gpytorch_model()
    
    @pytest.mark.skipif(True, reason="GPyTorch tests require complex setup")
    def test_create_gpytorch_model(self, skip_if_no_gpytorch):
        """Test GPyTorch model creation."""
        # This would require setting up training data first
        # Skipped for now due to complexity
        pass
    
    @pytest.mark.skipif(True, reason="GPyTorch tests require complex setup")
    def test_train_gpytorch_model(self, skip_if_no_gpytorch):
        """Test GPyTorch model training."""
        # This would require setting up training data first
        # Skipped for now due to complexity
        pass


class TestUnifiedSPE9ToolkitEvaluation:
    """Test model evaluation functionality."""
    
    def test_evaluate_model_not_found(self):
        """Test evaluate_model raises error for non-existent model."""
        toolkit = UnifiedSPE9Toolkit()
        with pytest.raises(ValueError, match="Model GPR not found"):
            toolkit.evaluate_model('GPR')
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_evaluate_sklearn_model(self, mock_load, sample_grdecl_data):
        """Test sklearn model evaluation."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        toolkit.load_data()
        toolkit.prepare_features()
        toolkit.create_train_test_split()
        toolkit.setup_scalers()
        
        model = toolkit.create_sklearn_model('gpr')
        toolkit.train_sklearn_model(model, 'GPR')
        
        results = toolkit.evaluate_model('GPR')
        
        assert 'r2' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'y_pred' in results


class TestUnifiedSPE9ToolkitIntegration:
    """Integration tests for UnifiedSPE9Toolkit."""
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_full_sklearn_workflow(self, mock_load, sample_grdecl_data):
        """Test complete sklearn workflow."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        
        # Complete workflow
        toolkit.load_data()
        toolkit.prepare_features()
        toolkit.create_train_test_split(test_size=0.2)
        toolkit.setup_scalers()
        
        gpr = toolkit.create_sklearn_model('gpr')
        toolkit.train_sklearn_model(gpr, 'GPR')
        results = toolkit.evaluate_model('GPR')
        
        # Check that everything worked
        assert 'r2' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'y_pred' in results
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_multiple_sklearn_models(self, mock_load, sample_grdecl_data):
        """Test training multiple sklearn models."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit(backend='sklearn')
        
        # Setup
        toolkit.load_data()
        toolkit.prepare_features()
        toolkit.create_train_test_split(test_size=0.2)
        toolkit.setup_scalers()
        
        # Train multiple models
        gpr = toolkit.create_sklearn_model('gpr')
        rf = toolkit.create_sklearn_model('rf')
        
        toolkit.train_sklearn_model(gpr, 'GPR')
        toolkit.train_sklearn_model(rf, 'RF')
        
        # Evaluate both
        gpr_results = toolkit.evaluate_model('GPR')
        rf_results = toolkit.evaluate_model('RF')
        
        assert len(toolkit.models) == 2
        assert len(toolkit.results) == 2
        assert 'r2' in gpr_results
        assert 'r2' in rf_results


class TestUnifiedSPE9ToolkitUtilities:
    """Test utility functions."""
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_log_transform_option(self, mock_load, sample_grdecl_data):
        """Test log transform option in train/test split."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        toolkit.load_data()
        toolkit.prepare_features()
        
        # Test with log transform
        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            log_transform=True
        )
        
        # y values should be log-transformed (all should be different from original)
        original_y = toolkit.y_grid[toolkit.valid_mask]
        assert not np.allclose(np.concatenate([y_train, y_test]), original_y)
    
    @patch('spe9_geomodeling.unified_toolkit.load_spe9_data')
    def test_train_size_option(self, mock_load, sample_grdecl_data):
        """Test train_size option in train/test split."""
        mock_load.return_value = sample_grdecl_data
        
        toolkit = UnifiedSPE9Toolkit()
        toolkit.load_data()
        toolkit.prepare_features()
        
        # Test with specific train size
        X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
            train_size=50
        )
        
        assert len(X_train) == 50
        assert len(y_train) == 50
