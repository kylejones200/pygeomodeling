# Advanced Features

This guide covers the advanced features added to PyGeomodeling for production use.

## Model Serialization

Save and load trained models with complete metadata and versioning.

### Basic Usage

```python
from spe9_geomodeling import save_model, load_model
from sklearn.gaussian_process import GaussianProcessRegressor

# Train your model
model = GaussianProcessRegressor()
model.fit(X_train, y_train)

# Save with metadata
save_model(
    model=model,
    model_name="my_gpr_model",
    model_type="gpr",
    backend="sklearn",
    save_dir="saved_models",
    scaler=scaler,  # Optional
    metrics={"r2": 0.85, "mse": 0.12},  # Optional
    description="Production model trained on SPE9 dataset"
)
```

### Loading Models

```python
# Load the model
model, metadata, scaler = load_model("my_gpr_model", save_dir="saved_models")

# Check metadata
print(f"Model: {metadata.model_name}")
print(f"Type: {metadata.model_type}")
print(f"Created: {metadata.created_at}")
print(f"Metrics: {metadata.performance_metrics}")

# Use the model
predictions = model.predict(X_test)
```

### Advanced Serialization

```python
from spe9_geomodeling import ModelSerializer, ModelMetadata

# Create serializer
serializer = ModelSerializer(save_dir="models")

# Create detailed metadata
metadata = ModelMetadata(
    model_name="production_gpr_v1",
    model_type="gpr",
    backend="sklearn",
    version="1.0"
)

# Add training information
metadata.add_training_info(
    n_samples=len(X_train),
    n_features=X_train.shape[1],
    feature_names=["x", "y", "z"],
    training_time=15.3
)

# Add hyperparameters
metadata.add_hyperparameters({
    "kernel": "RBF + Matern",
    "alpha": 1e-10,
    "n_restarts_optimizer": 10
})

# Save with full metadata
model_dir = serializer.save_model(model, metadata, scaler)

# List all saved models
models = serializer.list_models()
print(f"Saved models: {models}")

# Get model info without loading
info = serializer.get_model_info("production_gpr_v1")
```

## Spatial Cross-Validation

Proper cross-validation for spatial data that accounts for spatial autocorrelation.

### Spatial K-Fold

```python
from spe9_geomodeling import SpatialKFold, cross_validate_spatial
from sklearn.gaussian_process import GaussianProcessRegressor

# Create spatial CV splitter
cv = SpatialKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
model = GaussianProcessRegressor()
results = cross_validate_spatial(
    model=model,
    X=X,  # Must have spatial coordinates in first 3 columns
    y=y,
    cv=cv,
    scoring="r2",
    return_train_score=True,
    verbose=True
)

print(f"Test R²: {results['test_score'].mean():.4f} ± {results['test_score'].std():.4f}")
print(f"Train R²: {results['train_score'].mean():.4f} ± {results['train_score'].std():.4f}")
```

### Block Cross-Validation

```python
from spe9_geomodeling import BlockCV

# Create block CV with 3x3x1 blocks
cv = BlockCV(n_blocks_x=3, n_blocks_y=3, n_blocks_z=1, buffer_size=0.1)

# Use with cross-validation
results = cross_validate_spatial(model, X, y, cv=cv)
```

### Hyperparameter Tuning with Optuna

```python
from spe9_geomodeling import HyperparameterTuner
from sklearn.ensemble import RandomForestRegressor

# Define parameter search space
param_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 20},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
}

# Create tuner
tuner = HyperparameterTuner(
    model_class=RandomForestRegressor,
    param_space=param_space,
    cv=5,
    n_trials=100,
    scoring="r2",
    random_state=42
)

# Run tuning
results = tuner.tune(X, y, verbose=True)

# Get best model
best_model = tuner.get_best_model()
best_model.fit(X_train, y_train)

# Access optimization history
study = results["study"]
print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

## Parallel Processing

Speed up model training and predictions using parallel processing.

### Parallel Model Training

```python
from spe9_geomodeling import ParallelModelTrainer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Define multiple models
models = {
    "gpr_rbf": GaussianProcessRegressor(),
    "random_forest": RandomForestRegressor(n_estimators=100),
    "svr": SVR(kernel="rbf"),
}

# Train all models in parallel
trainer = ParallelModelTrainer(n_jobs=-1, verbose=1)
trained_models = trainer.train_models(models, X_train, y_train)

# Train and evaluate in parallel
results = trainer.train_and_evaluate(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# View results
for name, result in results.items():
    print(f"\n{name}:")
    print(f"  R²: {result['metrics']['r2']:.4f}")
    print(f"  MSE: {result['metrics']['mse']:.4f}")
    print(f"  Training time: {result['training_time']:.2f}s")
```

### Batch Predictions

```python
from spe9_geomodeling import BatchPredictor

# Create batch predictor
predictor = BatchPredictor(n_jobs=-1, batch_size=1000, verbose=True)

# Make predictions in parallel batches
predictions = predictor.predict(model, X_large_dataset)

# For GP models with uncertainty
predictions, std_devs = predictor.predict(gp_model, X, return_std=True)

# Predict with multiple models
predictions_dict = predictor.predict_multiple_models(trained_models, X)
```

### Parallel Grid Search

```python
from spe9_geomodeling import parallel_grid_search
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
}

# Run parallel grid search
results = parallel_grid_search(
    model_class=RandomForestRegressor,
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_jobs=-1,
    verbose=True
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

### Parallel Cross-Validation

```python
from spe9_geomodeling import ParallelCrossValidator, SpatialKFold

# Create parallel CV
cv_parallel = ParallelCrossValidator(n_jobs=-1, verbose=True)

# Run cross-validation with parallel fold evaluation
cv_splitter = SpatialKFold(n_splits=10)
results = cv_parallel.cross_validate(
    model=model,
    X=X,
    y=y,
    cv_splitter=cv_splitter
)

print(f"Mean score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
```

## Error Handling

All modules now include comprehensive error handling with helpful suggestions.

### Custom Exceptions

```python
from spe9_geomodeling import exceptions

try:
    parser = GRDECLParser("nonexistent.grdecl")
    data = parser.load_data()
except exceptions.DataLoadError as e:
    print(e)  # Includes helpful suggestion
```

### Common Error Scenarios

```python
# File not found
try:
    data = load_spe9_data("missing_file.grdecl")
except exceptions.DataLoadError as e:
    print(e.message)
    print(e.suggestion)

# Invalid format
try:
    parser = GRDECLParser("invalid.txt")
    data = parser.load_data()
except exceptions.FileFormatError as e:
    print(e)

# Property not found
try:
    slice_data = parser.get_property_slice("INVALID_PROP")
except exceptions.PropertyNotFoundError as e:
    print(e)

# Model not trained
try:
    predictions = model.predict(X_test)
except exceptions.ModelNotTrainedError as e:
    print(e.suggestion)
```

## Complete Workflow Example

```python
from spe9_geomodeling import (
    load_spe9_data,
    UnifiedSPE9Toolkit,
    SpatialKFold,
    HyperparameterTuner,
    ParallelModelTrainer,
    save_model,
    cross_validate_spatial,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
data = load_spe9_data()

# 2. Prepare features
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# 3. Hyperparameter tuning with spatial CV
param_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 300},
    "max_depth": {"type": "int", "low": 5, "high": 20},
}

tuner = HyperparameterTuner(
    model_class=RandomForestRegressor,
    param_space=param_space,
    cv=SpatialKFold(n_splits=5),
    n_trials=50,
    random_state=42
)

tuning_results = tuner.tune(X_train, y_train)
best_model = tuner.get_best_model()

# 4. Train multiple models in parallel
models = {
    "tuned_rf": best_model,
    "gpr": GaussianProcessRegressor(),
    "rf_default": RandomForestRegressor(),
}

trainer = ParallelModelTrainer(n_jobs=-1)
results = trainer.train_and_evaluate(
    models, X_train, y_train, X_test, y_test
)

# 5. Save best model
best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["r2"])
best_model_obj = results[best_name]["model"]

save_model(
    model=best_model_obj,
    model_name=f"production_{best_name}",
    model_type=best_name,
    backend="sklearn",
    metrics=results[best_name]["metrics"],
    hyperparameters=tuning_results["best_params"]
)

print(f"\n✓ Best model: {best_name}")
print(f"  R²: {results[best_name]['metrics']['r2']:.4f}")
print(f"  Saved to: production_{best_name}")
```

## GPU Support (GPyTorch Models)

For GPyTorch models, GPU acceleration is automatically used when available.

```python
import torch
from spe9_geomodeling import UnifiedSPE9Toolkit

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create toolkit with GPyTorch backend
toolkit = UnifiedSPE9Toolkit(backend="gpytorch")
toolkit.load_data()

# Models will automatically use GPU if available
model = toolkit.create_gpytorch_model("deep_gp")
toolkit.train_gpytorch_model(model, "deep_gp_gpu")
```

## Best Practices

1. **Always use spatial cross-validation** for geostatistical data
2. **Save models with metadata** for reproducibility
3. **Use parallel processing** for large datasets or multiple models
4. **Tune hyperparameters** with Optuna for optimal performance
5. **Handle exceptions** gracefully with try-except blocks
6. **Version your models** when deploying to production
7. **Document hyperparameters** in model metadata
8. **Use batch predictions** for very large datasets
