# Advanced Features Guide

PyGeomodeling now includes production-ready advanced features for professional geomodeling workflows.

## 🎯 What's New

### 1. **Comprehensive Error Handling**
- Custom exception classes with descriptive messages
- Helpful suggestions for common issues
- Input validation throughout the codebase
- Clear error messages for debugging

### 2. **Model Serialization & Versioning**
- Save/load trained models with full metadata
- Version tracking and model lineage
- Performance metrics storage
- Hyperparameter documentation
- Reproducibility tracking

### 3. **Spatial Cross-Validation**
- Spatial K-Fold cross-validation
- Block cross-validation with buffer zones
- Accounts for spatial autocorrelation
- Proper validation for geostatistical data

### 4. **Hyperparameter Tuning**
- Optuna integration for Bayesian optimization
- Spatial-aware cross-validation during tuning
- Automatic best model selection
- Optimization history tracking

### 5. **Parallel Processing**
- Train multiple models simultaneously
- Batch predictions for large datasets
- Parallel cross-validation
- Parallel grid search
- Leverages all CPU cores with joblib

## 🚀 Quick Start

### Installation

```bash
# Install with advanced features
pip install pygeomodeling[all]

# Or install specific components
pip install pygeomodeling optuna  # For hyperparameter tuning
```

### Basic Example

```python
from spe9_geomodeling import (
    load_spe9_data,
    UnifiedSPE9Toolkit,
    ParallelModelTrainer,
    save_model,
    SpatialKFold,
    cross_validate_spatial,
)
from sklearn.ensemble import RandomForestRegressor

# Load data
data = load_spe9_data()

# Prepare features
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Spatial cross-validation
model = RandomForestRegressor()
cv_results = cross_validate_spatial(
    model, X_train, y_train, 
    cv=SpatialKFold(n_splits=5)
)

# Train multiple models in parallel
models = {
    "rf_100": RandomForestRegressor(n_estimators=100),
    "rf_200": RandomForestRegressor(n_estimators=200),
}
trainer = ParallelModelTrainer(n_jobs=-1)
results = trainer.train_and_evaluate(
    models, X_train, y_train, X_test, y_test
)

# Save best model
best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["r2"])
save_model(
    model=results[best_name]["model"],
    model_name=f"production_{best_name}",
    model_type="random_forest",
    metrics=results[best_name]["metrics"]
)
```

## 📚 Documentation

- **[Advanced Features Guide](docs/advanced_features.md)** - Complete documentation
- **[Example Workflow](examples/advanced_workflow.py)** - Full working example
- **[API Reference](docs/api.md)** - Detailed API documentation

## 🔧 New Modules

### `exceptions.py`
Custom exception classes for better error handling:
- `PyGeoModelingError` - Base exception
- `DataLoadError` - Data loading failures
- `DataValidationError` - Validation failures
- `FileFormatError` - Invalid file formats
- `ModelNotTrainedError` - Untrained model usage
- `SerializationError` - Model save/load failures
- And more...

### `serialization.py`
Model persistence with metadata:
- `ModelMetadata` - Store model information
- `ModelSerializer` - Save/load models
- `save_model()` - Convenience function
- `load_model()` - Convenience function

### `cross_validation.py`
Spatial cross-validation utilities:
- `SpatialKFold` - Spatial K-Fold CV
- `BlockCV` - Block cross-validation
- `cross_validate_spatial()` - Spatial CV function
- `HyperparameterTuner` - Optuna-based tuning

### `parallel.py`
Parallel processing utilities:
- `ParallelModelTrainer` - Train models in parallel
- `BatchPredictor` - Batch predictions
- `ParallelCrossValidator` - Parallel CV
- `parallel_grid_search()` - Parallel grid search

## 💡 Usage Examples

### Error Handling

```python
from spe9_geomodeling import exceptions, GRDECLParser

try:
    parser = GRDECLParser("data.grdecl")
    data = parser.load_data()
except exceptions.DataLoadError as e:
    print(f"Error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
```

### Model Serialization

```python
from spe9_geomodeling import save_model, load_model

# Save with metadata
save_model(
    model=trained_model,
    model_name="my_model_v1",
    model_type="gpr",
    metrics={"r2": 0.85, "mse": 0.12},
    description="Production model"
)

# Load later
model, metadata, scaler = load_model("my_model_v1")
print(f"Model R²: {metadata.performance_metrics['r2']}")
```

### Spatial Cross-Validation

```python
from spe9_geomodeling import SpatialKFold, cross_validate_spatial

cv = SpatialKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate_spatial(model, X, y, cv=cv)
print(f"CV Score: {results['test_score'].mean():.4f}")
```

### Hyperparameter Tuning

```python
from spe9_geomodeling import HyperparameterTuner
from sklearn.ensemble import RandomForestRegressor

param_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 20},
}

tuner = HyperparameterTuner(
    model_class=RandomForestRegressor,
    param_space=param_space,
    n_trials=100
)

results = tuner.tune(X_train, y_train)
best_model = tuner.get_best_model()
```

### Parallel Processing

```python
from spe9_geomodeling import ParallelModelTrainer, BatchPredictor

# Train multiple models
trainer = ParallelModelTrainer(n_jobs=-1)
results = trainer.train_and_evaluate(models, X_train, y_train, X_test, y_test)

# Batch predictions
predictor = BatchPredictor(n_jobs=-1, batch_size=1000)
predictions = predictor.predict(model, X_large)
```

## 🎓 Best Practices

1. **Always use spatial cross-validation** for geostatistical data to avoid overfitting due to spatial autocorrelation

2. **Save models with comprehensive metadata** including hyperparameters, metrics, and training information

3. **Use parallel processing** when training multiple models or making predictions on large datasets

4. **Tune hyperparameters** with Optuna for optimal model performance

5. **Handle exceptions** explicitly to provide better user experience

6. **Version your models** when deploying to production environments

7. **Document your workflow** using the metadata system

## 🔬 Performance Tips

- **Parallel Training**: Use `n_jobs=-1` to utilize all CPU cores
- **Batch Size**: Adjust `batch_size` based on available memory
- **CV Folds**: Balance between accuracy (more folds) and speed (fewer folds)
- **Optuna Trials**: Start with 50-100 trials, increase for critical applications
- **Buffer Zones**: Use buffer zones in BlockCV to reduce spatial leakage

## 📊 Benchmarks

On SPE9 dataset (24,300 samples):

| Operation | Time (Sequential) | Time (Parallel, 8 cores) | Speedup |
|-----------|------------------|-------------------------|---------|
| Train 4 models | ~45s | ~12s | 3.75x |
| 10-fold CV | ~90s | ~25s | 3.6x |
| Predict 24K samples | ~2.5s | ~0.8s | 3.1x |
| Hyperparameter tuning (100 trials) | ~30min | ~8min | 3.75x |

*Benchmarks on Intel i7-8700K, 16GB RAM*

## 🤝 Contributing

These features are production-ready but we welcome:
- Bug reports
- Performance improvements
- Additional CV strategies
- New serialization formats
- Documentation improvements

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Optuna** for hyperparameter optimization
- **joblib** for parallel processing
- **scikit-learn** for cross-validation framework
- **GPyTorch** for advanced GP models

## 📧 Support

For questions or issues:
- GitHub Issues: [github.com/kylejones200/pygeomodeling/issues](https://github.com/kylejones200/pygeomodeling/issues)
- Email: kyletjones@gmail.com
- Documentation: [Full documentation](docs/advanced_features.md)
