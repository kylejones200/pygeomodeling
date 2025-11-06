"""GPyTorch models for geomodeling.

Defines Gaussian Process models optimized for reservoir property prediction.
"""

from __future__ import annotations

import gpytorch
import torch


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
        *,
        kernel_type: str = "combined",
        ard: bool = True,
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
        self.covar_module = self._create_kernel(kernel_type, input_dim, ard)

    def _create_kernel(
        self, kernel_type: str, input_dim: int, ard: bool
    ) -> gpytorch.kernels.Kernel:
        """Create the covariance kernel.

        Args:
            kernel_type: Type of kernel to create
            input_dim: Number of input dimensions
            ard: Whether to use ARD

        Returns:
            Configured kernel
        """
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


# Backward compatibility alias
GPModel = SPE9GPModel


def create_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.Likelihood | None = None,
    *,
    model_type: str = "standard",
    **kwargs,
) -> tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.Likelihood]:
    """Create GP models.

    Args:
        train_x: Training input features
        train_y: Training target values
        likelihood: Optional likelihood (creates Gaussian if None)
        model_type: Type of model ('standard' or 'deep')
        **kwargs: Additional arguments for model creation

    Returns:
        Tuple of (model, likelihood)
    """
    if likelihood is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if model_type == "standard":
        model = SPE9GPModel(train_x, train_y, likelihood, **kwargs)
    elif model_type == "deep":
        model = DeepGPModel(train_x, train_y, likelihood, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, likelihood


if __name__ == "__main__":
    print("SPE9 GPyTorch Models")
    print("Available models:")
    print("- SPE9GPModel: Standard GP with flexible kernels")
    print("- DeepGPModel: Deep GP with neural network features")
    print("- create_gp_model(): Factory function for easy model creation")
