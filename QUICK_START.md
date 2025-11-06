# Quick Start Guide

Get up and running with PyGeomodeling in 5 minutes!

## Installation

```bash
# Basic installation
pip install pygeomodeling

# With all features
pip install pygeomodeling[all]
```

## Your First Model

```python
from spe9_geomodeling import load_spe9_data, UnifiedSPE9Toolkit

# 1. Load sample data
data = load_spe9_data('data/sample_small.grdecl')

# 2. Prepare features
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# 3. Train model
model = toolkit.create_sklearn_model('gpr', kernel_type='rbf')
toolkit.train_sklearn_model(model, 'my_first_model')

# 4. Evaluate
results = toolkit.evaluate_model('my_first_model', X_test, y_test)
print(f"R² Score: {results['r2']:.4f}")
```

## Next Steps

### Try the Tutorials

```bash
# Start Jupyter
jupyter notebook examples/notebooks/
```

1. **Getting Started** - Learn the basics (15 min)
2. **Advanced Modeling** - Spatial CV, parallel training (20 min)

### Explore Advanced Features

```python
from spe9_geomodeling import (
    SpatialKFold,           # Spatial cross-validation
    ParallelModelTrainer,   # Train multiple models
    save_model,             # Save with metadata
    HyperparameterTuner,    # Optimize hyperparameters
)
```

### Run Example Workflows

```bash
# Advanced workflow with all features
python examples/advanced_workflow.py
```

## Common Tasks

### Load Your Own Data

```python
from spe9_geomodeling import GRDECLParser

parser = GRDECLParser('path/to/your/file.grdecl')
data = parser.load_data()
```

### Train Multiple Models in Parallel

```python
from spe9_geomodeling import ParallelModelTrainer
from sklearn.ensemble import RandomForestRegressor

models = {
    'rf_100': RandomForestRegressor(n_estimators=100),
    'rf_200': RandomForestRegressor(n_estimators=200),
}

trainer = ParallelModelTrainer(n_jobs=-1)
results = trainer.train_and_evaluate(
    models, X_train, y_train, X_test, y_test
)
```

### Save and Load Models

```python
from spe9_geomodeling import save_model, load_model

# Save
save_model(
    model=trained_model,
    model_name='production_v1',
    model_type='gpr',
    metrics={'r2': 0.85}
)

# Load
model, metadata, scaler = load_model('production_v1')
```

### Spatial Cross-Validation

```python
from spe9_geomodeling import SpatialKFold, cross_validate_spatial

cv = SpatialKFold(n_splits=5)
results = cross_validate_spatial(model, X, y, cv=cv)
print(f"CV Score: {results['test_score'].mean():.4f}")
```

## Getting Help

- **Documentation**: [docs/](docs/)
- **Tutorials**: [examples/notebooks/](examples/notebooks/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/kylejones200/pygeomodeling/issues)
- **Email**: <kyletjones@gmail.com>

## What's Next?

- ✓ **Completed Quick Start** - You're ready to build models!
- → **Try Tutorials** - Learn advanced features
- → **Read Documentation** - Deep dive into capabilities
- → **Join Community** - Contribute and share

Happy modeling! 🚀
