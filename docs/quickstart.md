# Quick Start Tutorial

Get up and running with the SPE9 Geomodeling Toolkit in just 5 minutes!

## 🚀 5-Minute Quick Start

### Step 1: Installation

```bash
pip install pygeomodeling[advanced]
```

### Step 2: Basic Usage

```python
from pygeomodeling import UnifiedSPE9Toolkit, load_spe9_data

# Load SPE9 dataset (or use synthetic data if not available)
try:
    data = load_spe9_data()
    print(f"✅ Loaded real SPE9 data: {data['grid_shape']} grid")
except FileNotFoundError:
    print("ℹ️ Using synthetic data for demonstration")
    data = None

# Create and configure toolkit
toolkit = UnifiedSPE9Toolkit()
if data:
    toolkit.load_spe9_data(data)
else:
    toolkit.load_synthetic_data(grid_size=(50, 50, 10))

# Create train/test split
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train a Gaussian Process model
model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
toolkit.train_sklearn_model(model, 'combined_gpr')

# Evaluate performance
results = toolkit.evaluate_model('combined_gpr', X_test, y_test)
print(f"Model R² Score: {results.r2:.4f}")
```

### Step 3: Run Deep GP Experiment

```python
from pygeomodeling import DeepGPExperiment

# Run comprehensive model comparison
experiment = DeepGPExperiment()
results = experiment.run_comparison_experiment()

# Results saved to 'deep_gp_comparison.png'
print("✅ Experiment complete! Check deep_gp_comparison.png for results")
```

## 📊 Common Workflows

### Workflow 1: Traditional Geomodeling Pipeline

```python
from pygeomodeling import UnifiedSPE9Toolkit, GRDECLParser

# 1. Load your GRDECL data
parser = GRDECLParser('path/to/your/data.grdecl')
data = parser.load_data()

# 2. Set up toolkit
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)

# 3. Prepare data
X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
    test_size=0.2,
    random_state=42
)

# 4. Train multiple models
models = {
    'rbf_gpr': toolkit.create_sklearn_model('gpr', kernel_type='rbf'),
    'matern_gpr': toolkit.create_sklearn_model('gpr', kernel_type='matern'),
    'combined_gpr': toolkit.create_sklearn_model('gpr', kernel_type='combined')
}

# Train all models
for name, model in models.items():
    toolkit.train_sklearn_model(model, name)
    results = toolkit.evaluate_model(name, X_test, y_test)
    print(f"{name}: R² = {results.r2:.4f}, RMSE = {results.rmse:.2f}")

# 5. Make predictions and visualize
predictions = toolkit.predict_full_grid('combined_gpr')
toolkit.plot_results('combined_gpr', save_path='results.png')
```

### Workflow 2: Deep GP Research

```python
from pygeomodeling import DeepGPExperiment
from pygeomodeling.experiments import DeepGPExperiment

# Create experiment with custom configuration
experiment = DeepGPExperiment()

# Run with custom parameters
results = experiment.run_comparison_experiment(
    train_samples=1000,
    test_samples=200,
    training_iterations=100
)

# Analyze results
best_model = max(results.keys(), key=lambda x: results[x]['metrics']['r2_score'])
print(f"Best performing model: {best_model}")
print(f"R² Score: {results[best_model]['metrics']['r2_score']:.4f}")

# Access detailed metrics
for model_name, model_results in results.items():
    metrics = model_results['metrics']
    print(f"{model_name}:")
    print(f"  R² = {metrics['r2_score']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.2f}")
    print(f"  Training time = {model_results['training_time']:.1f}s")
```

### Workflow 3: Custom Model Development

```python
from spe9_geomodeling import SPE9GPModel, create_gp_model
import torch
import gpytorch

# Create custom GP model
model = create_gp_model(
    model_type='standard',
    kernel_type='rbf',
    ard=True,  # Automatic Relevance Determination
    input_dim=3
)

# Or create Deep GP model
deep_model = create_gp_model(
    model_type='deep',
    hidden_dims=[50, 25],
    kernel_type='rbf',
    input_dim=3
)

# Custom training loop
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Training code here...
```

## 🎯 Key Concepts

### Data Loading

The toolkit supports multiple data sources:

- **SPE9 GRDECL files**: Real reservoir data
- **Synthetic data**: Generated for testing and development
- **Custom arrays**: Your own spatial data

### Model Types

Choose from various modeling approaches:

- **Traditional GP**: RBF, Matérn, Combined kernels
- **Deep GP**: Neural network feature extraction
- **Kriging**: Classical geostatistical methods

### Evaluation Metrics

Standard metrics for model assessment:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Training Time**: Computational efficiency

## 🔧 Configuration Options

### Toolkit Configuration

```python
toolkit = UnifiedSPE9Toolkit(
    backend='sklearn',  # or 'gpytorch'
    random_state=42,
    verbose=True
)
```

### Model Configuration

```python
# Scikit-learn models
model = toolkit.create_sklearn_model(
    model_type='gpr',
    kernel_type='combined',
    alpha=1e-10,
    n_restarts_optimizer=5
)

# GPyTorch models
model = toolkit.create_gpytorch_model(
    model_type='deep',
    hidden_dims=[64, 32],
    kernel_type='rbf',
    inducing_points=100
)
```

## 📈 Performance Tips

### For Large Datasets

```python
# Use sparse GP approximations
model = toolkit.create_gpytorch_model(
    model_type='sparse',
    inducing_points=500,  # Reduce for faster training
    kernel_type='rbf'
)

# Use smaller training sets
X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
    train_size=0.1,  # Use only 10% for training
    test_size=0.05
)
```

### For Better Accuracy

```python
# Use combined kernels
model = toolkit.create_sklearn_model(
    model_type='gpr',
    kernel_type='combined',  # RBF + Matérn
    n_restarts_optimizer=10  # More optimization restarts
)

# Use more training data
X_train, X_test, y_train, y_test = toolkit.create_train_test_split(
    train_size=0.8,  # Use 80% for training
    test_size=0.2
)
```

## 🐛 Common Issues

### Issue: Poor Model Performance

```python
# Check data distribution
import matplotlib.pyplot as plt
plt.hist(y_train, bins=50)
plt.title('Training Data Distribution')
plt.show()

# Try log transformation for skewed data
toolkit.apply_log_transform = True
```

### Issue: Slow Training

```python
# Reduce training set size
X_train_small = X_train[:500]  # Use first 500 samples
y_train_small = y_train[:500]

# Use faster kernels
model = toolkit.create_sklearn_model('gpr', kernel_type='rbf')  # Faster than combined
```

### Issue: Memory Errors

```python
# Use sparse approximations
model = toolkit.create_gpytorch_model(
    model_type='sparse',
    inducing_points=200  # Reduce memory usage
)
```

## 📚 Next Steps

Now that you're up and running:

1. **Explore Examples**: Check out `examples/` directory for more detailed workflows
2. **Read API Docs**: See [API Reference](api.md) for complete function documentation
3. **Model Comparison**: Learn about different approaches in [Model Comparison Guide](model_comparison.md)
4. **Advanced Features**: Dive into [Deep GP Experiments](deep_gp.md)

## 💡 Pro Tips

- **Start Simple**: Begin with traditional GP models before trying Deep GP
- **Validate Results**: Always check model performance on held-out test data
- **Visualize**: Use the built-in plotting functions to understand your results
- **Experiment**: Try different kernels and hyperparameters
- **Save Models**: Use `joblib` to save trained models for later use

---

**Ready to dive deeper?** Check out the [API Reference](api.md) for complete documentation of all available functions and classes.
