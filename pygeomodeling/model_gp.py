"""GPyTorch models for geomodeling.

Defines Gaussian Process models optimized for reservoir property prediction.
Includes multi-output GP, sparse GP, neural network GP, and Bayesian neural networks.
"""

from __future__ import annotations

import logging
from typing import Any

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Configure logging
logger = logging.getLogger(__name__)

# Import advanced kernels (optional)
try:
    from .advanced_kernels import NonStationaryRBFKernel, SpectralMixtureKernel

    ADVANCED_KERNELS_AVAILABLE = True
except ImportError:
    ADVANCED_KERNELS_AVAILABLE = False
    NonStationaryRBFKernel: type[Any] | None = None  # type: ignore[no-redef]
    SpectralMixtureKernel: type[Any] | None = None  # type: ignore[no-redef]


class SPE9GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model for SPE9 reservoir property prediction.

    This model uses a combination of RBF and Matérn kernels, which is well-suited
    for modeling spatial correlations in geological properties like permeability.

    Attributes:
        mean_module: Mean function (constant mean)
        covar_module: Covariance function (scaled RBF + Matérn kernel)
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel_type: str = "combined",
        ard: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the GP model.

        Args:
            train_x: Training input features [N, D]
            train_y: Training target values [N]
            likelihood: GPyTorch likelihood function
            kernel_type: Type of kernel ('rbf', 'matern', 'combined')
            ard: Whether to use Automatic Relevance Determination
        """
        super().__init__(train_x, train_y, likelihood)

        # Mean function - constant mean is appropriate for log-transformed data
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance function based on kernel type
        input_dim = train_x.shape[-1]
        use_advanced = kwargs.pop("use_advanced_kernels", False)
        self.covar_module = self._create_kernel(kernel_type, input_dim, ard, use_advanced)

    def _create_kernel(
        self, kernel_type: str, input_dim: int, ard: bool, use_advanced: bool = False
    ) -> gpytorch.kernels.Kernel:
        """Create the covariance kernel.

        Args:
            kernel_type: Type of kernel to create ('rbf', 'matern', 'combined',
                        'nonstationary_rbf', 'spectral_mixture', 'sm')
            input_dim: Number of input dimensions
            ard: Whether to use ARD (for standard kernels)
            use_advanced: Whether to use advanced kernels if available

        Returns:
            Configured kernel
        """
        # Check for advanced kernels
        if use_advanced and ADVANCED_KERNELS_AVAILABLE:
            if kernel_type in ["nonstationary_rbf"]:
                base_kernel = NonStationaryRBFKernel(input_dim=input_dim)
                return gpytorch.kernels.ScaleKernel(base_kernel)
            elif kernel_type in ["spectral_mixture", "sm"]:
                num_mixtures = 4  # Default number of mixtures
                base_kernel = SpectralMixtureKernel(
                    input_dim=input_dim, num_mixtures=num_mixtures
                )
                return base_kernel  # SM kernel handles scaling internally

        # Standard kernels
        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "combined":
            rbf_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
            base_kernel = rbf_kernel + matern_kernel
        else:
            raise ValueError(
                f"Unknown kernel_type: {kernel_type}. "
                "Choose from: 'rbf', 'matern', 'combined', 'nonstationary_rbf', 'spectral_mixture'"
            )

        # Scale kernel to learn output variance
        return gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the GP model.

        Args:
            x: Input features [N, D]

        Returns:
            Multivariate normal distribution over function values
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepGPModel(gpytorch.models.ExactGP):
    """Deep Gaussian Process model for more complex spatial patterns.

    This model uses a neural network feature extractor followed by a GP,
    which can capture more complex non-linear relationships in the data.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Initialize the Deep GP model.

        Args:
            train_x: Training input features [N, D]
            train_y: Training target values [N]
            likelihood: GPyTorch likelihood function
            hidden_dim: Hidden dimension size for neural network
            num_layers: Number of hidden layers
        """
        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[-1]

        # Neural network feature extractor
        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    torch.nn.Linear(current_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        self.feature_extractor = torch.nn.Sequential(*layers)

        # GP components
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the Deep GP model.

        Args:
            x: Input features [N, D]

        Returns:
            Multivariate normal distribution over function values
        """
        # Extract features using neural network
        features = self.feature_extractor(x)

        # Apply GP to extracted features
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiOutputGPModel(gpytorch.models.ExactGP):
    """Multi-output Gaussian Process for modeling multiple correlated spatial fields.

    This model uses a Linear Model of Coregionalization (LMC) to model correlations
    between multiple output variables, allowing information sharing across outputs.

    Args:
        train_x: Training input features [N, D]
        train_y: Training target values [N, num_outputs]
        likelihood: Multi-task likelihood
        num_outputs: Number of output dimensions
        kernel_type: Type of base kernel ('rbf', 'matern', 'combined')
        ard: Whether to use Automatic Relevance Determination
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        *,
        num_outputs: int = 2,
        kernel_type: str = "combined",
        ard: bool = True,
        rank: int | None = None,
    ) -> None:
        if train_y.shape[-1] != num_outputs:
            raise ValueError(
                f"train_y must have {num_outputs} output dimensions, "
                f"got {train_y.shape[-1]}"
            )

        super().__init__(train_x, train_y, likelihood)

        self.num_outputs = num_outputs
        input_dim = train_x.shape[-1]

        # Mean function (one per output)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_outputs
        )

        # Base kernel for each latent function
        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "combined":
            rbf_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
            base_kernel = rbf_kernel + matern_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        # Scale kernel
        scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)

        # Multi-task kernel using Linear Model of Coregionalization
        if rank is None:
            rank = min(num_outputs, 3)  # Default rank for LMC

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            scaled_kernel, num_tasks=num_outputs, rank=rank
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Forward pass through the multi-output GP model.

        Args:
            x: Input features [N, D]

        Returns:
            Multitask multivariate normal distribution [N, num_outputs]
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    """Sparse Gaussian Process for large-scale problems (>1M cells).

    Uses inducing points to approximate the full GP, reducing computational
    complexity from O(N^3) to O(M^2N) where M << N is the number of inducing points.

    Args:
        inducing_points: Initial inducing point locations [M, D]
        num_outputs: Number of output dimensions (1 for single output)
        kernel_type: Type of kernel ('rbf', 'matern', 'combined')
        ard: Whether to use ARD
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        *,
        num_outputs: int = 1,
        kernel_type: str = "combined",
        ard: bool = True,
    ) -> None:
        # Variational strategy
        if num_outputs == 1:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.shape[0]
            )
        else:
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
                inducing_points.shape[0]
            )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        if num_outputs > 1:
            variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        self.num_outputs = num_outputs
        input_dim = inducing_points.shape[-1]

        # Mean function
        if num_outputs == 1:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_outputs
            )

        # Kernel
        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
        elif kernel_type == "combined":
            rbf_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim if ard else None
            )
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=input_dim if ard else None
            )
            base_kernel = rbf_kernel + matern_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        if num_outputs > 1:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                self.covar_module, num_tasks=num_outputs
            )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the sparse GP model.

        Args:
            x: Input features [N, D]

        Returns:
            Multivariate normal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.num_outputs == 1:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class NeuralNetworkGPModel(gpytorch.models.ExactGP):
    """Enhanced Neural Network Gaussian Process with hybrid deep architectures.

    Combines neural network feature extraction with GP uncertainty quantification.
    Supports various architectures including residual connections and attention mechanisms.

    Args:
        train_x: Training input features [N, D]
        train_y: Training target values [N]
        likelihood: GPyTorch likelihood function
        architecture: NN architecture type ('standard', 'residual', 'attention')
        hidden_dim: Hidden dimension size
        num_layers: Number of hidden layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        *,
        architecture: str = "standard",
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[-1]
        self.architecture = architecture

        # Build feature extractor based on architecture
        if architecture == "standard":
            self.feature_extractor = self._build_standard_network(
                input_dim, hidden_dim, num_layers, dropout
            )
        elif architecture == "residual":
            self.feature_extractor = self._build_residual_network(
                input_dim, hidden_dim, num_layers, dropout
            )
        elif architecture == "attention":
            self.feature_extractor = self._build_attention_network(
                input_dim, hidden_dim, num_layers, dropout
            )
        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                "Choose from: 'standard', 'residual', 'attention'"
            )

        # GP components
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=hidden_dim)
        )

    def _build_standard_network(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Build standard feedforward network."""
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*layers)

    def _build_residual_network(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Build residual network with skip connections."""

        class ResidualBlock(nn.Module):
            def __init__(self, dim: int, dropout: float):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                )
                self.activation = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.activation(self.block(x) + x)

        layers = []
        # Initial projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout))

        return nn.Sequential(*layers)

    def _build_attention_network(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Build network with self-attention mechanism."""

        class AttentionBlock(nn.Module):
            def __init__(self, dim: int, dropout: float):
                super().__init__()
                self.attention = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
                self.ff = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Self-attention
                x_norm = self.norm1(x)
                attn_out, _ = self.attention(x_norm, x_norm, x_norm)
                x = x + attn_out

                # Feedforward
                x = x + self.ff(self.norm2(x))
                return x

        layers = []
        # Initial projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))

        # Attention blocks
        for _ in range(num_layers):
            layers.append(AttentionBlock(hidden_dim, dropout))

        # Final projection
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the neural network GP model.

        Args:
            x: Input features [N, D]

        Returns:
            Multivariate normal distribution over function values
        """
        # Extract features
        if self.architecture == "attention":
            # Attention expects [batch, seq_len, features]
            x_expanded = x.unsqueeze(1)  # [N, 1, D]
            features = self.feature_extractor(x_expanded)  # [N, 1, hidden_dim]
            features = features.squeeze(1)  # [N, hidden_dim]
        else:
            features = self.feature_extractor(x)  # [N, hidden_dim]

        # Apply GP to extracted features
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network with variational inference.

    Implements a neural network with Bayesian priors on weights, providing
    uncertainty quantification through approximate posterior inference.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension size
        num_layers: Number of hidden layers
        output_dim: Number of output dimensions
        prior_std: Standard deviation of weight priors
        variational_samples: Number of samples for variational inference
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1,
        prior_std: float = 1.0,
        variational_samples: int = 10,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        self.variational_samples = variational_samples

        # Build layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(
                self._create_variational_layer(current_dim, hidden_dim, f"layer_{i}")
            )
            current_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.output_layer = self._create_variational_layer(
            hidden_dim, output_dim, "output"
        )
        self.activation = nn.ReLU()

    def _create_variational_layer(
        self, in_dim: int, out_dim: int, name: str
    ) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        """Create a variational layer with mean and log-variance parameters."""
        # Mean and log-variance for weights
        weight_mu = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        weight_logvar = nn.Parameter(torch.randn(out_dim, in_dim) * -2.0)

        # Mean and log-variance for biases
        bias_mu = nn.Parameter(torch.zeros(out_dim))
        bias_logvar = nn.Parameter(torch.zeros(out_dim) - 2.0)

        # Register as attributes for parameter tracking
        self.register_parameter(f"{name}_weight_mu", weight_mu)
        self.register_parameter(f"{name}_weight_logvar", weight_logvar)
        self.register_parameter(f"{name}_bias_mu", bias_mu)
        self.register_parameter(f"{name}_bias_logvar", bias_logvar)

        return (weight_mu, weight_logvar, bias_mu, bias_logvar)

    def _sample_weights(
        self, mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample weights from variational posterior."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + std.unsqueeze(0) * eps

    def forward(
        self, x: torch.Tensor, return_std: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with variational sampling.

        Args:
            x: Input features [N, D]
            return_std: If True, return standard deviation of predictions

        Returns:
            Predictions [N, output_dim] or (mean, std) if return_std=True
        """
        num_samples = self.variational_samples if self.training or return_std else 1
        predictions = []

        for _ in range(num_samples):
            h = x
            # Forward through hidden layers
            for layer_params in self.layers:
                weight_mu, weight_logvar, bias_mu, bias_logvar = layer_params
                W = self._sample_weights(weight_mu, weight_logvar, 1).squeeze(0)
                b = self._sample_weights(bias_mu, bias_logvar, 1).squeeze(0)
                h = F.linear(h, W, b)
                h = self.activation(h)

            # Output layer
            weight_mu, weight_logvar, bias_mu, bias_logvar = self.output_layer
            W = self._sample_weights(weight_mu, weight_logvar, 1).squeeze(0)
            b = self._sample_weights(bias_mu, bias_logvar, 1).squeeze(0)
            pred = F.linear(h, W, b)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_samples, N, output_dim]

        if return_std:
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return predictions.mean(dim=0)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational posterior and prior.

        Returns:
            Total KL divergence (scalar)
        """
        kl = 0.0
        prior_std_sq = self.prior_std**2

        # KL for all layers
        for layer_params in self.layers:
            _, weight_logvar, _, bias_logvar = layer_params
            weight_mu = layer_params[0]
            bias_mu = layer_params[2]

            # Weight KL
            weight_var = torch.exp(weight_logvar)
            kl += 0.5 * (
                torch.sum(weight_var / prior_std_sq)
                + torch.sum(weight_mu**2 / prior_std_sq)
                - torch.sum(weight_logvar)
                - weight_mu.numel() * torch.log(torch.tensor(prior_std_sq))
            )

            # Bias KL
            bias_var = torch.exp(bias_logvar)
            kl += 0.5 * (
                torch.sum(bias_var / prior_std_sq)
                + torch.sum(bias_mu**2 / prior_std_sq)
                - torch.sum(bias_logvar)
                - bias_mu.numel() * torch.log(torch.tensor(prior_std_sq))
            )

        # Output layer KL
        weight_mu, weight_logvar, bias_mu, bias_logvar = self.output_layer
        weight_var = torch.exp(weight_logvar)
        kl += 0.5 * (
            torch.sum(weight_var / prior_std_sq)
            + torch.sum(weight_mu**2 / prior_std_sq)
            - torch.sum(weight_logvar)
            - weight_mu.numel() * torch.log(torch.tensor(prior_std_sq))
        )

        bias_var = torch.exp(bias_logvar)
        kl += 0.5 * (
            torch.sum(bias_var / prior_std_sq)
            + torch.sum(bias_mu**2 / prior_std_sq)
            - torch.sum(bias_logvar)
            - bias_mu.numel() * torch.log(torch.tensor(prior_std_sq))
        )

        return kl


# Backward compatibility alias
GPModel: type[Any] = SPE9GPModel


def create_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.Likelihood | None = None,
    *,
    model_type: str = "standard",
    **kwargs,
) -> tuple[gpytorch.models.ExactGP | gpytorch.models.ApproximateGP, gpytorch.likelihoods.Likelihood]:
    """Create GP models.

    Args:
        train_x: Training input features
        train_y: Training target values (can be [N] or [N, num_outputs] for multi-output)
        likelihood: Optional likelihood (creates Gaussian if None)
        model_type: Type of model ('standard', 'deep', 'multioutput', 'sparse', 'neural')
        **kwargs: Additional arguments for model creation

    Returns:
        Tuple of (model, likelihood)

    Examples:
        >>> # Standard GP
        >>> model, likelihood = create_gp_model(X, y, model_type='standard')
        >>> # Multi-output GP
        >>> model, likelihood = create_gp_model(X, y_multi, model_type='multioutput', num_outputs=3)
        >>> # Sparse GP
        >>> model, likelihood = create_gp_model(X, y, model_type='sparse', inducing_points=inducing_X)
        >>> # Neural Network GP
        >>> model, likelihood = create_gp_model(X, y, model_type='neural', architecture='residual')
    """
    if likelihood is None:
        # Check if multi-output
        if train_y.dim() > 1 and train_y.shape[-1] > 1:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=train_y.shape[-1]
            )
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if model_type == "standard":
        model = SPE9GPModel(train_x, train_y, likelihood, **kwargs)
    elif model_type == "deep":
        model = DeepGPModel(train_x, train_y, likelihood, **kwargs)
    elif model_type == "multioutput":
        if train_y.dim() == 1:
            raise ValueError("For multi-output GP, train_y must be 2D [N, num_outputs]")
        num_outputs = kwargs.pop("num_outputs", train_y.shape[-1])
        model = MultiOutputGPModel(train_x, train_y, likelihood, num_outputs=num_outputs, **kwargs)
    elif model_type == "sparse":
        inducing_points = kwargs.pop("inducing_points", None)
        if inducing_points is None:
            # Default: use k-means to select inducing points
            from sklearn.cluster import KMeans

            num_inducing = kwargs.pop("num_inducing", min(500, len(train_x)))
            kmeans = KMeans(n_clusters=num_inducing, random_state=42, n_init=10)
            kmeans.fit(train_x.cpu().numpy())
            inducing_points = torch.tensor(
                kmeans.cluster_centers_, dtype=train_x.dtype, device=train_x.device
            )
        num_outputs = kwargs.pop("num_outputs", 1)
        model = SparseGPModel(inducing_points, num_outputs=num_outputs, **kwargs)
    elif model_type == "neural":
        model = NeuralNetworkGPModel(train_x, train_y, likelihood, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose from: 'standard', 'deep', 'multioutput', 'sparse', 'neural'"
        )

    return model, likelihood


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("SPE9 GPyTorch Models")
    logger.info("Available models:")
    logger.info("- SPE9GPModel: Standard GP with flexible kernels")
    logger.info("- DeepGPModel: Deep GP with neural network features")
    logger.info("- MultiOutputGPModel: Multi-output GP for correlated fields")
    logger.info("- SparseGPModel: Sparse GP for large-scale problems")
    logger.info("- NeuralNetworkGPModel: Hybrid neural network GP")
    logger.info("- BayesianNeuralNetwork: Bayesian NN with uncertainty")
    logger.info("- create_gp_model(): Factory function for easy model creation")
