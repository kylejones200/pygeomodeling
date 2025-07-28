# Examples and Tutorials

This guide provides practical examples and step-by-step tutorials for using the SPE9 Geomodeling Toolkit.

## 🚀 Basic Examples

### Example 1: Simple GP Modeling
```python
from spe9_geomodeling import UnifiedSPE9Toolkit, load_spe9_data

# Load data
data = load_spe9_data()  # Uses default SPE9 path
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)

# Create train/test split
X_train, X_test, y_train, y_test = toolkit.create_train_test_split(test_size=0.2)

# Train RBF kernel GP
model = toolkit.create_sklearn_model('gpr', kernel_type='rbf')
toolkit.train_sklearn_model(model, 'rbf_gpr')

# Evaluate
results = toolkit.evaluate_model('rbf_gpr', X_test, y_test)
print(f"RBF GP R² Score: {results.r2:.4f}")
```

### Example 2: Model Comparison
```python
from spe9_geomodeling import UnifiedSPE9Toolkit
import matplotlib.pyplot as plt

toolkit = UnifiedSPE9Toolkit()
toolkit.load_synthetic_data(grid_size=(30, 30, 5))  # Synthetic for demo

X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Train multiple models
models = {
    'RBF GP': ('gpr', {'kernel_type': 'rbf'}),
    'Matérn GP': ('gpr', {'kernel_type': 'matern'}),
    'Combined GP': ('gpr', {'kernel_type': 'combined'}),
    'Random Forest': ('rf', {'n_estimators': 100})
}

results = {}
for name, (model_type, params) in models.items():
    model = toolkit.create_sklearn_model(model_type, **params)
    toolkit.train_sklearn_model(model, name.lower().replace(' ', '_'))
    results[name] = toolkit.evaluate_model(name.lower().replace(' ', '_'), X_test, y_test)

# Plot comparison
names = list(results.keys())
r2_scores = [results[name].r2 for name in names]

plt.figure(figsize=(10, 6))
plt.bar(names, r2_scores)
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Example 3: Deep GP Experiment
```python
from spe9_geomodeling import DeepGPExperiment

# Run comprehensive experiment
experiment = DeepGPExperiment()
results = experiment.run_comparison_experiment(
    train_samples=800,
    test_samples=200,
    training_iterations=75
)

# Analyze results
print("Deep GP Experiment Results:")
print("-" * 50)
for model_name, result in results.items():
    metrics = result['metrics']
    time = result['training_time']
    print(f"{model_name:20s} | R²: {metrics['r2_score']:.4f} | "
          f"RMSE: {metrics['rmse']:.2f} | Time: {time:.1f}s")

# Find best model
best_model = max(results.keys(), key=lambda x: results[x]['metrics']['r2_score'])
print(f"\nBest performing model: {best_model}")
```

## 📊 Advanced Examples

### Example 4: Custom Kernel Development
```python
from sklearn.gaussian_process.kernels import Kernel
import numpy as np

class CustomSpatialKernel(Kernel):
    """Custom kernel for geological data with anisotropy."""
    
    def __init__(self, length_scale_x=1.0, length_scale_y=1.0, length_scale_z=1.0):
        self.length_scale_x = length_scale_x
        self.length_scale_y = length_scale_y
        self.length_scale_z = length_scale_z
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        
        # Anisotropic distance calculation
        X_scaled = X / np.array([self.length_scale_x, self.length_scale_y, self.length_scale_z])
        Y_scaled = Y / np.array([self.length_scale_x, self.length_scale_y, self.length_scale_z])
        
        dists = np.sum((X_scaled[:, None] - Y_scaled[None, :]) ** 2, axis=2)
        K = np.exp(-0.5 * dists)
        
        if eval_gradient:
            # Gradient computation...
            return K, np.zeros((X.shape[0], Y.shape[0], 3))
        return K

# Use custom kernel
from sklearn.gaussian_process import GaussianProcessRegressor

custom_kernel = CustomSpatialKernel(length_scale_x=2.0, length_scale_y=2.0, length_scale_z=0.5)
model = GaussianProcessRegressor(kernel=custom_kernel)
model.fit(X_train, y_train)
```

### Example 5: Uncertainty Quantification
```python
from spe9_geomodeling import UnifiedSPE9Toolkit
import numpy as np
import matplotlib.pyplot as plt

toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(load_spe9_data())

# Train GP model
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()
model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
toolkit.train_sklearn_model(model, 'combined_gpr')

# Get predictions with uncertainty
predictions, uncertainties = toolkit.predict_with_uncertainty('combined_gpr', X_test)

# Analyze uncertainty
print("Uncertainty Analysis:")
print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
print(f"Max uncertainty: {np.max(uncertainties):.4f}")
print(f"Min uncertainty: {np.min(uncertainties):.4f}")

# Plot prediction vs uncertainty
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')

plt.subplot(1, 2, 2)
plt.scatter(predictions, uncertainties, alpha=0.6)
plt.xlabel('Predictions')
plt.ylabel('Uncertainty')
plt.title('Prediction Uncertainty')

plt.tight_layout()
plt.show()
```

### Example 6: Hyperparameter Optimization
```python
from spe9_geomodeling import UnifiedSPE9Toolkit
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern
import numpy as np

toolkit = UnifiedSPE9Toolkit()
toolkit.load_synthetic_data()

X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Define parameter grid
param_grid = {
    'alpha': [1e-10, 1e-8, 1e-6],
    'kernel__length_scale': np.logspace(-2, 2, 5),
    'kernel__k1__length_scale': np.logspace(-2, 2, 3),  # For combined kernels
    'kernel__k2__length_scale': np.logspace(-2, 2, 3),
}

# Create base model
base_model = toolkit.create_sklearn_model('gpr', kernel_type='combined')

# Grid search
grid_search = GridSearchCV(
    base_model, 
    param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("Running hyperparameter optimization...")
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print(f"Test set R² score: {test_score:.4f}")
```

## 🔬 Research Examples

### Example 7: Spatial Cross-Validation
```python
from sklearn.model_selection import GroupKFold
import numpy as np

def create_spatial_groups(X, n_groups=5):
    """Create spatial groups for cross-validation."""
    # Simple spatial grouping based on x-coordinate
    x_coords = X[:, 0]
    group_boundaries = np.percentile(x_coords, np.linspace(0, 100, n_groups + 1))
    groups = np.digitize(x_coords, group_boundaries) - 1
    return np.clip(groups, 0, n_groups - 1)

# Load data
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(load_spe9_data())
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Create spatial groups
spatial_groups = create_spatial_groups(X_train, n_groups=5)

# Spatial cross-validation
group_kfold = GroupKFold(n_splits=5)
model = toolkit.create_sklearn_model('gpr', kernel_type='combined')

spatial_scores = []
for train_idx, val_idx in group_kfold.split(X_train, y_train, groups=spatial_groups):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    score = model.score(X_fold_val, y_fold_val)
    spatial_scores.append(score)

print(f"Spatial CV scores: {spatial_scores}")
print(f"Mean spatial CV score: {np.mean(spatial_scores):.4f} ± {np.std(spatial_scores):.4f}")
```

### Example 8: Multi-Property Modeling
```python
# Simulate multi-property data
def create_multi_property_data(X):
    """Create correlated properties (PERMX, PORO, NTG)."""
    np.random.seed(42)
    
    # Base permeability
    permx = np.exp(np.random.normal(5, 2, len(X)))
    
    # Correlated porosity
    poro = 0.3 * np.log(permx) + np.random.normal(0, 0.05, len(X))
    poro = np.clip(poro, 0.05, 0.35)
    
    # Net-to-gross ratio
    ntg = 0.8 + 0.1 * np.log(permx) / np.max(np.log(permx)) + np.random.normal(0, 0.1, len(X))
    ntg = np.clip(ntg, 0.3, 1.0)
    
    return {'PERMX': permx, 'PORO': poro, 'NTG': ntg}

# Multi-property modeling
toolkit = UnifiedSPE9Toolkit()
toolkit.load_synthetic_data()
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Create multi-property targets
properties = create_multi_property_data(X_train)

# Train separate models for each property
models = {}
results = {}

for prop_name, prop_values in properties.items():
    print(f"\nTraining model for {prop_name}...")
    
    model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
    model.fit(X_train, prop_values)
    models[prop_name] = model
    
    # Evaluate on test set
    test_properties = create_multi_property_data(X_test)
    test_score = model.score(X_test, test_properties[prop_name])
    results[prop_name] = test_score
    
    print(f"{prop_name} R² score: {test_score:.4f}")

# Cross-property correlation analysis
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
prop_names = list(properties.keys())

for i, (prop1, prop2) in enumerate([(0,1), (0,2), (1,2)]):
    axes[i].scatter(properties[prop_names[prop1]], properties[prop_names[prop2]], alpha=0.6)
    axes[i].set_xlabel(prop_names[prop1])
    axes[i].set_ylabel(prop_names[prop2])
    
    # Calculate correlation
    corr = np.corrcoef(properties[prop_names[prop1]], properties[prop_names[prop2]])[0,1]
    axes[i].set_title(f'Correlation: {corr:.3f}')

plt.tight_layout()
plt.show()
```

## 🎨 Visualization Examples

### Example 9: Advanced Plotting
```python
from spe9_geomodeling import SPE9Plotter
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
toolkit = UnifiedSPE9Toolkit()
data = load_spe9_data()
toolkit.load_spe9_data(data)

# Train model
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()
model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
toolkit.train_sklearn_model(model, 'combined_gpr')

# Get full grid predictions
predictions = toolkit.predict_full_grid('combined_gpr')

# Create plotter
plotter = SPE9Plotter()

# Plot original vs predicted slices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original data slices
original_data = data['properties']['PERMX'].reshape(data['grid_shape'])
slice_indices = [5, 12, 20]  # Different Z slices

for i, z_idx in enumerate(slice_indices):
    # Original
    im1 = axes[0, i].imshow(original_data[:, :, z_idx], cmap='viridis', origin='lower')
    axes[0, i].set_title(f'Original PERMX - Z={z_idx}')
    plt.colorbar(im1, ax=axes[0, i])
    
    # Predicted
    pred_data = predictions.reshape(data['grid_shape'])
    im2 = axes[1, i].imshow(pred_data[:, :, z_idx], cmap='viridis', origin='lower')
    axes[1, i].set_title(f'Predicted PERMX - Z={z_idx}')
    plt.colorbar(im2, ax=axes[1, i])

plt.tight_layout()
plt.show()
```

### Example 10: Interactive Visualization
```python
# Requires: pip install plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_interactive_3d_plot(data, predictions):
    """Create interactive 3D visualization."""
    
    # Sample points for visualization (full grid would be too dense)
    n_points = 1000
    indices = np.random.choice(len(data), n_points, replace=False)
    
    x, y, z = data[indices, 0], data[indices, 1], data[indices, 2]
    values = predictions[indices]
    
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='Viridis',
            colorbar=dict(title="PERMX"),
            opacity=0.8
        ),
        text=[f'PERMX: {v:.2f}' for v in values],
        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='3D Permeability Distribution',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    return fig

# Create interactive plot
if 'predictions' in locals():
    fig = create_interactive_3d_plot(X_test, predictions[:len(X_test)])
    fig.show()
```

## 🔧 Integration Examples

### Example 11: Custom Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create custom preprocessing pipeline
def create_geomodeling_pipeline():
    """Create a complete geomodeling pipeline."""
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
        ('gpr', toolkit.create_sklearn_model('gpr', kernel_type='combined'))
    ])
    
    return pipeline

# Use pipeline
toolkit = UnifiedSPE9Toolkit()
toolkit.load_synthetic_data()
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

pipeline = create_geomodeling_pipeline()
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Pipeline R² score: {score:.4f}")

# Analyze components
print(f"PCA components retained: {pipeline.named_steps['pca'].n_components_}")
print(f"Explained variance ratio: {pipeline.named_steps['pca'].explained_variance_ratio_.sum():.4f}")
```

### Example 12: Export and Visualization
```python
from spe9_geomodeling import GRDECLParser

# Complete workflow with export
def complete_geomodeling_workflow(input_file, output_prefix):
    """Complete workflow from GRDECL input to results export."""
    
    # Load data
    parser = GRDECLParser(input_file)
    data = parser.load_data()
    
    # Set up toolkit
    toolkit = UnifiedSPE9Toolkit()
    toolkit.load_spe9_data(data)
    
    # Train model
    X_train, X_test, y_train, y_test = toolkit.create_train_test_split()
    model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
    toolkit.train_sklearn_model(model, 'final_model')
    
    # Evaluate
    results = toolkit.evaluate_model('final_model', X_test, y_test)
    print(f"Model Performance: R² = {results.r2:.4f}, RMSE = {results.rmse:.2f}")
    
    # Full grid prediction
    predictions = toolkit.predict_full_grid('final_model')
    uncertainties = toolkit.predict_uncertainty('final_model')
    
    # Export results
    toolkit.export_to_grdecl(predictions, f"{output_prefix}_predictions.grdecl")
    toolkit.export_to_grdecl(uncertainties, f"{output_prefix}_uncertainty.grdecl")
    
    # Create visualizations
    toolkit.plot_results('final_model', save_path=f"{output_prefix}_results.png")
    
    return results, predictions, uncertainties

# Run complete workflow
# results, preds, uncert = complete_geomodeling_workflow('SPE9.GRDECL', 'spe9_output')
```

## 💡 Tips and Best Practices

### Performance Tips
```python
# For large datasets
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)

# Use smaller training sets
X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
    train_size=0.1,  # Use only 10% for training
    test_size=0.05   # 5% for testing
)

# Use sparse GP for very large datasets
if len(X_train) > 5000:
    model = toolkit.create_gpytorch_model('sparse', inducing_points=500)
else:
    model = toolkit.create_sklearn_model('gpr', kernel_type='rbf')
```

### Memory Management
```python
import gc
import torch

# Clear GPU memory (if using GPyTorch)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Clear Python memory
gc.collect()

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")
```

### Error Handling
```python
from spe9_geomodeling import SPE9DataError, ModelTrainingError

try:
    toolkit = UnifiedSPE9Toolkit()
    data = load_spe9_data()
    toolkit.load_spe9_data(data)
    
    # Training...
    model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
    toolkit.train_sklearn_model(model, 'test_model')
    
except SPE9DataError as e:
    print(f"Data loading error: {e}")
    # Fallback to synthetic data
    toolkit.load_synthetic_data()
    
except ModelTrainingError as e:
    print(f"Training error: {e}")
    # Try simpler model
    model = toolkit.create_sklearn_model('gpr', kernel_type='rbf')
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error and continue with default settings
```

---

**Next Steps:**
- Explore [Deep GP Guide](deep_gp.md) for advanced modeling techniques
- Check [API Reference](api.md) for complete function documentation
- See [Model Comparison Guide](model_comparison.md) for choosing the right approach
