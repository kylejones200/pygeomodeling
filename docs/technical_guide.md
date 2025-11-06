# Technical Guide: Gaussian Processes and Kriging for Reservoir Modeling

*A practical guide to spatial interpolation with uncertainty quantification*

## Overview

Reservoir modeling requires accurate spatial estimates of rock properties. Oil and gas production forecasts, recovery strategies, and capital allocation decisions all depend on how well we understand the permeability field. But we rarely observe permeability directly across the entire volume—we measure it at sparse well locations.

**The Challenge**: Construct a gridded model of reservoir properties that:
- Honors observed data at well locations
- Provides reliable estimates elsewhere
- Quantifies uncertainty for decision-making
- Handles non-stationary spatial patterns

**The Solution**: Gaussian Process (GP) regression provides a flexible, probabilistic framework that addresses all these requirements.

## Theoretical Foundation

### Why Gaussian Processes?

Traditional interpolation methods have limitations:

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| **Linear Interpolation** | Fast, simple | No uncertainty, ignores spatial structure |
| **Inverse Distance** | Intuitive | Arbitrary power parameter, no uncertainty |
| **Kriging** | Spatial covariance | Assumes stationarity, requires variogram |
| **Gaussian Processes** | Flexible, probabilistic | Computational cost for large datasets |

**GP Advantages**:
1. **Probabilistic**: Returns mean AND variance
2. **Flexible**: Custom kernels for different spatial patterns
3. **Principled**: Bayesian framework
4. **Extensible**: Easy to add features beyond coordinates

### Mathematical Framework

A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

**Definition**:
```
f(x) ~ GP(m(x), k(x, x'))
```

Where:
- `m(x)` is the mean function (often constant)
- `k(x, x')` is the covariance (kernel) function

**Prediction**:
Given training data `(X, y)` and test points `X*`:

```
f* | X, y, X* ~ N(μ*, Σ*)

μ* = K(X*, X)[K(X, X) + σ²I]⁻¹ y
Σ* = K(X*, X*) - K(X*, X)[K(X, X) + σ²I]⁻¹ K(X, X*)
```

Where `K` is the kernel matrix and `σ²` is noise variance.

### Kernel Selection

The kernel encodes assumptions about spatial correlation:

**1. Radial Basis Function (RBF)**
```python
k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
```
- Infinitely differentiable
- Smooth predictions
- Good for continuous properties

**2. Matérn Kernel**
```python
k(x, x') = σ² (2^(1-ν)/Γ(ν)) (√(2ν)||x-x'||/ℓ)^ν K_ν(√(2ν)||x-x'||/ℓ)
```
- Controls smoothness via ν parameter
- ν = 0.5: Exponential (rough)
- ν = 1.5: Once differentiable
- ν = 2.5: Twice differentiable
- ν → ∞: RBF

**3. Composite Kernels**
```python
k_total = k_RBF + k_Matern  # Additive
k_total = k_RBF × k_Matern  # Multiplicative
```

## Implementation with PyGeomodeling

### 1. Data Preparation

```python
from spe9_geomodeling import GRDECLParser, UnifiedSPE9Toolkit
import numpy as np

# Load SPE9 data
parser = GRDECLParser('data/SPE9.GRDECL')
data = parser.load_data()

# Extract permeability
permx = data['properties']['PERMX']
nx, ny, nz = data['dimensions']

print(f"Grid: {nx} × {ny} × {nz} = {nx*ny*nz} cells")
print(f"PERMX range: [{permx.min():.2f}, {permx.max():.2f}] mD")
```

**Key Steps**:
1. Filter unrealistic values (zeros, negatives)
2. Log-transform for normality
3. Create coordinate arrays
4. Normalize coordinates

```python
# Create coordinates
x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

coordinates = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
values = permx.ravel()

# Filter valid values
mask = values > 0.1  # Remove unrealistic values
coordinates = coordinates[mask]
values = values[mask]

# Log transform
log_values = np.log(values)

print(f"Valid samples: {len(values)}")
```

### 2. Variogram Analysis

Before fitting a GP, analyze spatial correlation structure:

```python
from spe9_geomodeling import (
    compute_experimental_variogram,
    fit_variogram_model,
    plot_variogram
)

# Compute experimental variogram (2D slice for visualization)
layer = 5
coords_2d = coordinates[coordinates[:, 2] == layer][:, :2]
vals_2d = log_values[coordinates[:, 2] == layer]

lags, semi_var, n_pairs = compute_experimental_variogram(
    coords_2d, vals_2d, n_lags=15
)

# Fit spherical model
vario_model = fit_variogram_model(
    lags, semi_var, 
    model_type='spherical',
    weights=np.sqrt(n_pairs)
)

print(f"Nugget: {vario_model.nugget:.4f}")
print(f"Sill: {vario_model.sill:.4f}")
print(f"Range: {vario_model.range_param:.2f}")

# Visualize
plot_variogram(lags, semi_var, model=vario_model, n_pairs=n_pairs)
```

**Interpretation**:
- **Nugget**: Measurement error or micro-scale variability
- **Sill**: Total variance (should match sample variance)
- **Range**: Distance of spatial influence (~correlation length)

### 3. Gaussian Process Model

#### Using scikit-learn (Smaller Datasets)

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    coordinates, log_values, test_size=0.2, random_state=42
)

# Normalize coordinates
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define composite kernel
kernel = (
    ConstantKernel(1.0) * RBF(length_scale=1.0) +
    ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) +
    WhiteKernel(noise_level=0.1)
)

# Create and train GP
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42,
    normalize_y=True
)

print("Training GP model...")
gp.fit(X_train_scaled, y_train)

print(f"Optimized kernel: {gp.kernel_}")
print(f"Log-likelihood: {gp.log_marginal_likelihood_value_:.2f}")
```

#### Using GPyTorch (Larger Datasets)

```python
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + 
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Convert to tensors
train_x = torch.FloatTensor(X_train_scaled)
train_y = torch.FloatTensor(y_train)

# Initialize model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print("Training GPyTorch model...")
for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 10 == 0:
        print(f"Iter {i+1}/50 - Loss: {loss.item():.3f}")
```

### 4. Prediction with Uncertainty

```python
# Predict on test set
y_pred, y_std = gp.predict(X_test_scaled, return_std=True)

# Back-transform from log space
perm_pred = np.exp(y_pred)
perm_true = np.exp(y_test)

# Compute metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(perm_true, perm_pred)
rmse = np.sqrt(mean_squared_error(perm_true, perm_pred))
mae = mean_absolute_error(perm_true, perm_pred)

print(f"\nTest Set Performance:")
print(f"  R²:   {r2:.4f}")
print(f"  RMSE: {rmse:.2f} mD")
print(f"  MAE:  {mae:.2f} mD")

# Uncertainty statistics
print(f"\nUncertainty (log space):")
print(f"  Mean std: {y_std.mean():.4f}")
print(f"  Min std:  {y_std.min():.4f}")
print(f"  Max std:  {y_std.max():.4f}")
```

### 5. Full Field Prediction

```python
# Predict across entire grid
print("\nPredicting full field...")
X_full_scaled = scaler.transform(coordinates)

# Predict in batches for memory efficiency
from spe9_geomodeling import BatchPredictor

predictor = BatchPredictor(batch_size=5000, n_jobs=-1)
y_full_pred, y_full_std = predictor.predict(
    gp, X_full_scaled, return_std=True
)

# Reshape to 3D grid
perm_pred_3d = np.exp(y_full_pred).reshape(nx, ny, nz)
perm_std_3d = y_full_std.reshape(nx, ny, nz)

print(f"Predicted field shape: {perm_pred_3d.shape}")
```

### 6. Visualization

```python
import matplotlib.pyplot as plt

# Plot predictions vs truth
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, layer_idx in enumerate([0, nz//2, nz-1]):
    # Predicted
    im1 = axes[0, i].imshow(
        perm_pred_3d[:, :, layer_idx].T,
        cmap='viridis', origin='lower'
    )
    axes[0, i].set_title(f'Predicted - Layer {layer_idx}')
    plt.colorbar(im1, ax=axes[0, i], label='Permeability (mD)')
    
    # Uncertainty
    im2 = axes[1, i].imshow(
        perm_std_3d[:, :, layer_idx].T,
        cmap='Reds', origin='lower'
    )
    axes[1, i].set_title(f'Uncertainty - Layer {layer_idx}')
    plt.colorbar(im2, ax=axes[1, i], label='Std Dev (log space)')

plt.tight_layout()
plt.savefig('gp_predictions.png', dpi=300, bbox_inches='tight')
```

### 7. Export for Simulation

```python
# Export to GRDECL format
from spe9_geomodeling import GRDECLParser

def export_property_to_grdecl(values_3d, property_name, filename):
    """Export 3D property array to GRDECL format"""
    with open(filename, 'w') as f:
        f.write(f"{property_name}\n")
        
        # Flatten in Fortran order (column-major)
        values_flat = values_3d.ravel(order='F')
        
        # Write values (8 per line)
        for i in range(0, len(values_flat), 8):
            line_values = values_flat[i:i+8]
            line = ' '.join(f'{v:.6f}' for v in line_values)
            f.write(f"  {line}\n")
        
        f.write("/\n")

# Export predicted permeability
export_property_to_grdecl(perm_pred_3d, 'PERMX', 'PERMX_GPR.GRDECL')

# Export uncertainty
export_property_to_grdecl(perm_std_3d, 'SIGMA', 'SIGMA_GPR.GRDECL')

print("✓ Exported to GRDECL format for Eclipse/CMG")
```

## Comparison: GP vs Kriging

### Ordinary Kriging Implementation

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Kriging is equivalent to GP with specific kernel
# Use variogram range as length scale
kriging_kernel = RBF(length_scale=vario_model.range_param)

kriging_model = GaussianProcessRegressor(
    kernel=kriging_kernel,
    alpha=vario_model.nugget,  # Nugget as noise
    optimizer=None,  # Don't optimize (use variogram params)
    normalize_y=False
)

kriging_model.fit(X_train_scaled, y_train)
y_krig_pred = kriging_model.predict(X_test_scaled)

# Compare
print("\nKriging vs GP:")
print(f"  Kriging R²: {r2_score(y_test, y_krig_pred):.4f}")
print(f"  GP R²:      {r2:.4f}")
```

### Key Differences

| Aspect | Kriging | Gaussian Process |
|--------|---------|------------------|
| **Kernel** | From variogram | Learned from data |
| **Optimization** | Manual (variogram) | Automatic (MLE) |
| **Flexibility** | Limited to spatial | Can add features |
| **Dimensionality** | Typically 2D | Any dimension |
| **Uncertainty** | Yes | Yes |
| **Stationarity** | Assumed | Can handle non-stationary |

## Advanced Topics

### 1. Sparse GP for Large Datasets

```python
import gpytorch

class SparseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Use 500 inducing points for 10,000+ training samples
n_inducing = 500
inducing_indices = np.random.choice(len(X_train), n_inducing, replace=False)
inducing_points = torch.FloatTensor(X_train_scaled[inducing_indices])

sparse_model = SparseGPModel(train_x, train_y, likelihood, inducing_points)
```

### 2. Multi-Output GP (Co-Kriging)

```python
# Predict multiple properties simultaneously
class MultiOutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Train on PERMX and PORO simultaneously
```

### 3. Anisotropic Kernels

```python
# Different length scales in different directions
class AnisotropicRBF(gpytorch.kernels.Kernel):
    def __init__(self, ard_num_dims):
        super().__init__()
        self.register_parameter(
            'lengthscale',
            torch.nn.Parameter(torch.ones(ard_num_dims))
        )
    
    def forward(self, x1, x2):
        # Scale each dimension independently
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        dist = torch.cdist(x1_scaled, x2_scaled)
        return torch.exp(-0.5 * dist ** 2)

# Use for reservoirs with different correlation lengths in x, y, z
aniso_kernel = AnisotropicRBF(ard_num_dims=3)
```

## Best Practices

### 1. Data Preparation
- ✓ Remove outliers and invalid values
- ✓ Log-transform skewed distributions
- ✓ Normalize coordinates
- ✓ Check for trends (detrend if needed)

### 2. Model Selection
- ✓ Start with variogram analysis
- ✓ Use composite kernels (RBF + Matérn)
- ✓ Cross-validate with spatial folds
- ✓ Compare multiple models

### 3. Validation
- ✓ Hold-out test set
- ✓ Spatial cross-validation
- ✓ Check uncertainty calibration
- ✓ Visual inspection of predictions

### 4. Computational Efficiency
- ✓ Use sparse methods for >5,000 points
- ✓ Batch predictions for memory
- ✓ Parallel processing where possible
- ✓ GPU acceleration for GPyTorch

### 5. Production Deployment
- ✓ Save models with metadata
- ✓ Version control predictions
- ✓ Export to standard formats
- ✓ Document assumptions and limitations

## Limitations and Considerations

### Computational Complexity
- **Exact GP**: O(n³) for training, O(n²) for prediction
- **Solution**: Sparse methods, inducing points, or local GP

### Stationarity Assumptions
- **Issue**: Kernel assumes similar correlation everywhere
- **Solution**: Detrend data, use local GPs, or non-stationary kernels

### Kernel Selection
- **Challenge**: Many kernel options
- **Solution**: Start with variogram, use composite kernels, cross-validate

### Extrapolation
- **Issue**: GPs revert to prior mean outside data range
- **Solution**: Ensure good data coverage, use physical constraints

## Conclusion

Gaussian Process regression provides a solid foundation for modern geomodeling. It offers:

1. **Mathematical rigor**: Bayesian framework with clear assumptions
2. **Uncertainty quantification**: Built-in confidence intervals
3. **Flexibility**: Custom kernels for different spatial patterns
4. **Extensibility**: Easy to add features beyond coordinates

When paired with variogram analysis, spatial cross-validation, and proper visualization, GPs become a powerful tool for reservoir engineers seeking interpretable and defensible models.

**Key Takeaway**: Move beyond basic interpolation toward a more rigorous, probabilistic understanding of the reservoir.

## References

- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
- Chilès & Delfiner (2012). *Geostatistics: Modeling Spatial Uncertainty*
- SPE9 Benchmark: [Society of Petroleum Engineers](https://www.spe.org/)
- GPyTorch Documentation: [gpytorch.ai](https://gpytorch.ai/)

---

**Next Steps**:
- Try the tutorial notebooks in `examples/notebooks/`
- Explore the variogram analysis module
- Experiment with different kernels
- Apply to your own reservoir data
