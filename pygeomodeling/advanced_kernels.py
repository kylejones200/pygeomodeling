"""Advanced kernel functions for Gaussian Process models.

This module provides non-stationary kernels and spectral mixture kernels
for capturing complex spatial patterns in geostatistical data.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import gpytorch
    import torch
    import torch.nn.functional as F
    from gpytorch.constraints import Positive
    from gpytorch.kernels import Kernel
    from gpytorch.priors import NormalPrior

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    # Create dummy classes for type hints
    Kernel = object  # type: ignore[assignment,misc]
    Positive = None  # type: ignore[assignment,misc]
    NormalPrior = None  # type: ignore[assignment,misc]
    gpytorch = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

if not GPYTORCH_AVAILABLE:
    logger.warning(
        "GPyTorch not available. Advanced kernels require gpytorch. "
        "Install with: pip install pygeomodeling[advanced]"
    )


if GPYTORCH_AVAILABLE:

    class NonStationaryRBFKernel(Kernel):
        pass
    """Non-stationary RBF kernel with spatially varying length scales.

    The length scale is modeled as a function of input location using a GP,
    allowing the kernel to adapt to local spatial structure.

    Args:
        input_dim: Number of input dimensions
        lengthscale_prior: Prior for length scale parameters
        active_dims: Which dimensions to use
    """

    def __init__(
        self,
        input_dim: int,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
        active_dims: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.active_dims = active_dims

        # Length scale function parameters (GP over input space)
        # Using a simple parameterization: lengthscale = base + variation * sigmoid(linear_transform)
        lengthscale_constraint = Positive()
        if lengthscale_prior is None:
            lengthscale_prior = NormalPrior(loc=1.0, scale=1.0)

        self.register_parameter(
            name="raw_base_lengthscale",
            parameter=torch.nn.Parameter(torch.ones(input_dim)),
        )
        self.register_prior(
            "lengthscale_prior",
            lengthscale_prior,
            lambda: self.base_lengthscale,
            lambda v: self._set_base_lengthscale(v),
        )
        self.register_constraint("raw_base_lengthscale", lengthscale_constraint)

        # Variation amplitude (use softplus to ensure positivity)
        self.register_parameter(
            name="raw_lengthscale_variation",
            parameter=torch.nn.Parameter(torch.zeros(input_dim)),
        )

        # Linear transform for spatial variation
        self.register_parameter(
            name="raw_spatial_transform",
            parameter=torch.nn.Parameter(torch.zeros(input_dim, input_dim)),
        )

    @property
    def base_lengthscale(self) -> torch.Tensor:
        return self.raw_base_lengthscale_constraint.transform(self.raw_base_lengthscale)

    @base_lengthscale.setter
    def base_lengthscale(self, value: torch.Tensor) -> None:
        self._set_base_lengthscale(value)

    def _set_base_lengthscale(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_base_lengthscale)
        self.initialize(raw_base_lengthscale=self.raw_base_lengthscale_constraint.inverse_transform(value))

    @property
    def lengthscale_variation(self) -> torch.Tensor:
        return self.raw_lengthscale_variation_constraint.transform(self.raw_lengthscale_variation)

    @property
    def lengthscale_variation(self) -> torch.Tensor:
        """Length scale variation amplitude (always positive via softplus)."""
        return F.softplus(self.raw_lengthscale_variation)

    def _compute_lengthscale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatially varying length scale.

        Args:
            x: Input locations [N, D]

        Returns:
            Length scales [N, D]
        """
        # Transform input
        transformed = x @ torch.tanh(self.raw_spatial_transform)
        # Spatial variation component
        variation = self.lengthscale_variation * torch.sigmoid(transformed.sum(dim=-1, keepdim=True))
        # Base + variation
        lengthscale = self.base_lengthscale + variation.expand(-1, x.shape[-1])
        return lengthscale.clamp(min=1e-6)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        """Compute covariance matrix.

        Args:
            x1: First input [N, D]
            x2: Second input [M, D]
            diag: If True, return diagonal only
            **params: Additional parameters

        Returns:
            Covariance matrix [N, M] or diagonal [N] if diag=True
        """
        if diag:
            return torch.ones(x1.shape[0], device=x1.device, dtype=x1.dtype)

        # Compute length scales for each input
        lengthscale1 = self._compute_lengthscale(x1)  # [N, D]
        lengthscale2 = self._compute_lengthscale(x2)  # [M, D]

        # Average length scale (geometric mean)
        avg_lengthscale = torch.sqrt(lengthscale1.unsqueeze(1) * lengthscale2.unsqueeze(0))  # [N, M, D]

        # Distance scaled by length scale
        x1_ = x1.unsqueeze(1) / avg_lengthscale  # [N, 1, D] / [N, M, D] -> [N, M, D]
        x2_ = x2.unsqueeze(0) / avg_lengthscale  # [1, M, D] / [N, M, D] -> [N, M, D]
        dist = (x1_ - x2_).pow(2).sum(dim=-1)  # [N, M]

        return torch.exp(-0.5 * dist)

else:

    class NonStationaryRBFKernel:  # type: ignore[no-redef]
        """Placeholder for Non-Stationary RBF Kernel when GPyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GPyTorch is required for NonStationaryRBFKernel. "
                "Install with: pip install pygeomodeling[advanced]"
            )


if GPYTORCH_AVAILABLE:

    class SpectralMixtureKernel(Kernel):
        pass
    """Spectral Mixture (SM) kernel for flexible pattern learning.

    The SM kernel is derived from Bochner's theorem and can approximate
    any stationary kernel by learning a mixture of Q Gaussian components
    in the spectral domain.

    Args:
        input_dim: Number of input dimensions
        num_mixtures: Number of mixture components Q
        active_dims: Which dimensions to use
    """

    def __init__(
        self,
        input_dim: int,
        num_mixtures: int = 4,
        active_dims: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_mixtures = num_mixtures
        self.active_dims = active_dims

        # Mixture weights (must sum to 1)
        self.register_parameter(
            name="raw_mixture_weights",
            parameter=torch.nn.Parameter(torch.ones(num_mixtures) / num_mixtures),
        )

        # Mean frequencies for each mixture [Q, D]
        mean_freq_prior = NormalPrior(loc=0.0, scale=1.0)
        self.register_parameter(
            name="raw_mean_frequencies",
            parameter=torch.nn.Parameter(torch.randn(num_mixtures, input_dim) * 0.1),
        )
        self.register_prior(
            "mean_frequencies_prior",
            mean_freq_prior,
            lambda: self.mean_frequencies,
            lambda v: self._set_mean_frequencies(v),
        )

        # Length scales (inverse bandwidths) for each mixture [Q, D]
        lengthscale_constraint = Positive()
        lengthscale_prior = NormalPrior(loc=1.0, scale=1.0)
        self.register_parameter(
            name="raw_mixture_lengthscales",
            parameter=torch.nn.Parameter(torch.ones(num_mixtures, input_dim)),
        )
        self.register_prior(
            "mixture_lengthscales_prior",
            lengthscale_prior,
            lambda: self.mixture_lengthscales,
            lambda v: self._set_mixture_lengthscales(v),
        )
        self.register_constraint("raw_mixture_lengthscales", lengthscale_constraint)

    @property
    def mixture_weights(self) -> torch.Tensor:
        """Normalized mixture weights."""
        weights = torch.softmax(self.raw_mixture_weights, dim=0)
        return weights

    @property
    def mean_frequencies(self) -> torch.Tensor:
        return self.raw_mean_frequencies

    @mean_frequencies.setter
    def mean_frequencies(self, value: torch.Tensor) -> None:
        self._set_mean_frequencies(value)

    def _set_mean_frequencies(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mean_frequencies)
        self.initialize(raw_mean_frequencies=value)

    @property
    def mixture_lengthscales(self) -> torch.Tensor:
        return self.raw_mixture_lengthscales_constraint.transform(self.raw_mixture_lengthscales)

    @mixture_lengthscales.setter
    def mixture_lengthscales(self, value: torch.Tensor) -> None:
        self._set_mixture_lengthscales(value)

    def _set_mixture_lengthscales(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_lengthscales)
        self.initialize(
            raw_mixture_lengthscales=self.raw_mixture_lengthscales_constraint.inverse_transform(value)
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        """Compute spectral mixture covariance.

        Args:
            x1: First input [N, D]
            x2: Second input [M, D]
            diag: If True, return diagonal only
            **params: Additional parameters

        Returns:
            Covariance matrix [N, M] or diagonal [N] if diag=True
        """
        if diag:
            return self.mixture_weights.sum() * torch.ones(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )

        # Compute difference
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # [N, M, D]

        # Compute covariance for each mixture component
        covariances = []
        weights = self.mixture_weights  # [Q]

        for q in range(self.num_mixtures):
            # Get parameters for this mixture
            mean_freq = self.mean_frequencies[q]  # [D]
            lengthscale = self.mixture_lengthscales[q]  # [D]

            # RBF component in spectral domain
            diff_scaled = diff / lengthscale.unsqueeze(0).unsqueeze(0)  # [N, M, D]
            rbf_term = torch.exp(-0.5 * (diff_scaled**2).sum(dim=-1))  # [N, M]

            # Cosine component from mean frequency
            freq_term = torch.cos(2 * torch.pi * (diff * mean_freq.unsqueeze(0).unsqueeze(0)).sum(dim=-1))  # [N, M]

            # Mixture component
            cov_q = weights[q] * rbf_term * freq_term  # [N, M]
            covariances.append(cov_q)

        # Sum over mixtures
        return torch.stack(covariances, dim=0).sum(dim=0)  # [N, M]

else:

    class SpectralMixtureKernel:  # type: ignore[no-redef]
        """Placeholder for Spectral Mixture Kernel when GPyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GPyTorch is required for SpectralMixtureKernel. "
                "Install with: pip install pygeomodeling[advanced]"
            )


def create_advanced_kernel(
    kernel_type: str,
    input_dim: int,
    **kwargs,
) -> Kernel:
    """Factory function for creating advanced kernels.

    Args:
        kernel_type: Type of kernel ('nonstationary_rbf', 'spectral_mixture', 'sm')
        input_dim: Number of input dimensions
        **kwargs: Additional kernel-specific parameters

    Returns:
        Configured kernel

    Examples:
        >>> kernel = create_advanced_kernel('spectral_mixture', input_dim=3, num_mixtures=4)
        >>> kernel = create_advanced_kernel('nonstationary_rbf', input_dim=3)
    """
    if kernel_type in ["spectral_mixture", "sm"]:
        num_mixtures = kwargs.pop("num_mixtures", 4)
        return SpectralMixtureKernel(input_dim=input_dim, num_mixtures=num_mixtures, **kwargs)
    elif kernel_type == "nonstationary_rbf":
        return NonStationaryRBFKernel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown advanced kernel type: {kernel_type}. "
            "Choose from: 'spectral_mixture', 'sm', 'nonstationary_rbf'"
        )

