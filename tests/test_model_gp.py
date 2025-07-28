"""Tests for GP model functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Skip all tests if GPyTorch is not available
pytest_plugins = []
try:
    import torch
    import gpytorch

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False


@pytest.mark.skipif(not GPYTORCH_AVAILABLE, reason="GPyTorch not available")
class TestSPE9GPModel:
    """Test cases for SPE9GPModel class."""

    def test_model_initialization(self):
        """Test GP model initialization."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        # Create sample data
        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)

        assert model.mean_module is not None
        assert model.covar_module is not None
        assert hasattr(model, "forward")

    def test_model_initialization_with_kernel_types(self):
        """Test GP model initialization with different kernel types."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Test different kernel types
        for kernel_type in ["rbf", "matern", "combined"]:
            model = SPE9GPModel(train_x, train_y, likelihood, kernel_type=kernel_type)
            assert model.covar_module is not None

    def test_model_forward(self):
        """Test GP model forward pass."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)

        # Test forward pass
        test_x = torch.randn(10, 3)
        with torch.no_grad():
            output = model(test_x)
            assert isinstance(output, gpytorch.distributions.MultivariateNormal)
            assert output.mean.shape == (10,)

    def test_model_ard_option(self):
        """Test Automatic Relevance Determination option."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 5)  # 5 features
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Test with ARD
        model_ard = SPE9GPModel(train_x, train_y, likelihood, ard=True)

        # Test without ARD
        model_no_ard = SPE9GPModel(train_x, train_y, likelihood, ard=False)

        # Both should work
        assert model_ard.covar_module is not None
        assert model_no_ard.covar_module is not None


@pytest.mark.skipif(not GPYTORCH_AVAILABLE, reason="GPyTorch not available")
class TestDeepGPModel:
    """Test cases for DeepGPModel class."""

    def test_deep_model_initialization(self):
        """Test Deep GP model initialization."""
        from spe9_geomodeling.model_gp import DeepGPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = DeepGPModel(train_x, train_y, likelihood)

        assert model.feature_extractor is not None
        assert model.mean_module is not None
        assert model.covar_module is not None

    def test_deep_model_with_custom_architecture(self):
        """Test Deep GP model with custom architecture."""
        from spe9_geomodeling.model_gp import DeepGPModel

        train_x = torch.randn(50, 5)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = DeepGPModel(train_x, train_y, likelihood, hidden_dim=32, num_layers=3)

        assert model.feature_extractor is not None
        # Check that the network has the right structure
        layers = list(model.feature_extractor.children())
        # Should have 3 layers * 3 components each (Linear, ReLU, Dropout)
        assert len(layers) == 9

    def test_deep_model_forward(self):
        """Test Deep GP model forward pass."""
        from spe9_geomodeling.model_gp import DeepGPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = DeepGPModel(train_x, train_y, likelihood)

        # Test forward pass
        test_x = torch.randn(10, 3)
        with torch.no_grad():
            output = model(test_x)
            assert isinstance(output, gpytorch.distributions.MultivariateNormal)
            assert output.mean.shape == (10,)


@pytest.mark.skipif(not GPYTORCH_AVAILABLE, reason="GPyTorch not available")
class TestCreateGPModel:
    """Test cases for create_gp_model function."""

    def test_create_standard_gp_model(self):
        """Test creating standard GP model."""
        from spe9_geomodeling.model_gp import create_gp_model

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)

        model, likelihood = create_gp_model(train_x, train_y, model_type="standard")

        assert model is not None
        assert likelihood is not None
        assert isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)

    def test_create_deep_gp_model(self):
        """Test creating Deep GP model."""
        from spe9_geomodeling.model_gp import create_gp_model

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)

        model, likelihood = create_gp_model(train_x, train_y, model_type="deep")

        assert model is not None
        assert likelihood is not None
        assert hasattr(model, "feature_extractor")

    def test_create_gp_model_with_options(self):
        """Test creating GP model with custom options."""
        from spe9_geomodeling.model_gp import create_gp_model

        train_x = torch.randn(50, 5)
        train_y = torch.randn(50)

        model, likelihood = create_gp_model(
            train_x, train_y, model_type="standard", kernel_type="matern", ard=True
        )

        assert model is not None
        assert likelihood is not None

    def test_create_gp_model_invalid_type(self):
        """Test creating GP model with invalid type."""
        from spe9_geomodeling.model_gp import create_gp_model

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)

        with pytest.raises(ValueError, match="Unknown model_type"):
            create_gp_model(train_x, train_y, model_type="invalid")


@pytest.mark.skipif(not GPYTORCH_AVAILABLE, reason="GPyTorch not available")
class TestGPModelTraining:
    """Test GP model training functionality."""

    def test_model_training_mode(self):
        """Test model training mode switching."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)

        # Test training mode
        model.train()
        likelihood.train()
        assert model.training
        assert likelihood.training

        # Test eval mode
        model.eval()
        likelihood.eval()
        assert not model.training
        assert not likelihood.training

    def test_model_parameters(self):
        """Test model parameters are accessible."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)

        # Check that model has parameters
        params = list(model.parameters())
        assert len(params) > 0

        # Check that likelihood has parameters
        likelihood_params = list(likelihood.parameters())
        assert len(likelihood_params) > 0


@pytest.mark.skipif(not GPYTORCH_AVAILABLE, reason="GPyTorch not available")
class TestGPModelPrediction:
    """Test GP model prediction functionality."""

    def test_model_prediction_shapes(self):
        """Test prediction output shapes."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)
        model.eval()
        likelihood.eval()

        # Test prediction
        test_x = torch.randn(20, 3)
        with torch.no_grad():
            f_pred = model(test_x)
            y_pred = likelihood(f_pred)

            assert f_pred.mean.shape == (20,)
            assert f_pred.variance.shape == (20,)
            assert y_pred.mean.shape == (20,)
            assert y_pred.variance.shape == (20,)

    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        from spe9_geomodeling.model_gp import SPE9GPModel

        train_x = torch.randn(50, 3)
        train_y = torch.randn(50)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = SPE9GPModel(train_x, train_y, likelihood)
        model.eval()
        likelihood.eval()

        # Test different batch sizes
        for batch_size in [1, 10, 100]:
            test_x = torch.randn(batch_size, 3)
            with torch.no_grad():
                output = model(test_x)
                assert output.mean.shape == (batch_size,)


class TestGPModelWithoutGPyTorch:
    """Test behavior when GPyTorch is not available."""

    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # This test runs regardless of GPyTorch availability
        try:
            from spe9_geomodeling.model_gp import SPE9GPModel

            # If import succeeds, GPyTorch is available
            assert GPYTORCH_AVAILABLE
        except ImportError:
            # If import fails, GPyTorch is not available
            assert not GPYTORCH_AVAILABLE
