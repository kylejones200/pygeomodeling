# API Reference

Complete API documentation for the SPE9 Geomodeling Toolkit.

## 📦 Main Package: `spe9_geomodeling`

### Core Classes

#### `UnifiedSPE9Toolkit`
The main interface for geomodeling workflows supporting both scikit-learn and GPyTorch backends.

```python
from spe9_geomodeling import UnifiedSPE9Toolkit

toolkit = UnifiedSPE9Toolkit(backend='sklearn', random_state=42, verbose=True)
```

**Parameters:**
- `backend` (str): Backend to use ('sklearn' or 'gpytorch'). Default: 'sklearn'
- `random_state` (int): Random seed for reproducibility. Default: 42
- `verbose` (bool): Enable verbose output. Default: False

**Methods:**

##### `load_spe9_data(data: dict)`
Load SPE9 dataset into the toolkit.

**Parameters:**
- `data` (dict): Dictionary containing grid data and properties

**Example:**
```python
from spe9_geomodeling import load_spe9_data
data = load_spe9_data()
toolkit.load_spe9_data(data)
```

##### `load_synthetic_data(grid_size: tuple = (50, 50, 10))`
Generate and load synthetic spatial data for testing.

**Parameters:**
- `grid_size` (tuple): Dimensions of synthetic grid (nx, ny, nz)

##### `create_train_test_split(test_size: float = 0.2, train_size: float = None, random_state: int = None)`
Create train/test split from loaded data.

**Parameters:**
- `test_size` (float): Fraction of data for testing (0.0-1.0)
- `train_size` (float): Fraction of data for training (optional)
- `random_state` (int): Random seed (uses toolkit default if None)

**Returns:**
- `tuple`: (X_train, X_test, y_train, y_test)

##### `create_sklearn_model(model_type: str, **kwargs)`
Create a scikit-learn based model.

**Parameters:**
- `model_type` (str): Type of model ('gpr', 'rf', 'svr')
- `kernel_type` (str): For GPR models ('rbf', 'matern', 'combined')
- `**kwargs`: Additional model parameters

**Returns:**
- Model instance

**Example:**
```python
model = toolkit.create_sklearn_model('gpr', kernel_type='combined', alpha=1e-10)
```

##### `create_gpytorch_model(model_type: str, **kwargs)`
Create a GPyTorch based model.

**Parameters:**
- `model_type` (str): Type of model ('standard', 'deep', 'sparse')
- `kernel_type` (str): Kernel type ('rbf', 'matern', 'spectral_mixture')
- `hidden_dims` (list): For deep models, hidden layer dimensions
- `**kwargs**: Additional model parameters

**Returns:**
- GPyTorch model instance

##### `train_sklearn_model(model, model_name: str)`
Train a scikit-learn model.

**Parameters:**
- `model`: Model instance to train
- `model_name` (str): Name to store the trained model

##### `train_gpytorch_model(model, likelihood, model_name: str, training_iter: int = 50)`
Train a GPyTorch model.

**Parameters:**
- `model`: GPyTorch model instance
- `likelihood`: GPyTorch likelihood
- `model_name` (str): Name to store the trained model
- `training_iter` (int): Number of training iterations

##### `evaluate_model(model_name: str, X_test, y_test)`
Evaluate a trained model.

**Parameters:**
- `model_name` (str): Name of trained model
- `X_test`: Test features
- `y_test`: Test targets

**Returns:**
- `EvaluationResults`: Object with r2, rmse, mae attributes

##### `predict_full_grid(model_name: str)`
Make predictions on the full spatial grid.

**Parameters:**
- `model_name` (str): Name of trained model

**Returns:**
- `numpy.ndarray`: Predictions for all grid points

---

#### `GRDECLParser`
Parser for Eclipse GRDECL files.

```python
from spe9_geomodeling import GRDECLParser

parser = GRDECLParser('path/to/file.grdecl')
```

**Parameters:**
- `file_path` (str): Path to GRDECL file

**Methods:**

##### `load_data()`
Load and parse the GRDECL file.

**Returns:**
- `dict`: Dictionary containing grid dimensions, coordinates, and properties

##### `get_property(property_name: str)`
Extract a specific property from the loaded data.

**Parameters:**
- `property_name` (str): Name of property to extract

**Returns:**
- `numpy.ndarray`: Property values

---

#### `SPE9Plotter`
Visualization utilities for SPE9 data and model results.

```python
from spe9_geomodeling import SPE9Plotter

plotter = SPE9Plotter()
```

**Methods:**

##### `plot_property_slices(data: dict, property_name: str, save_path: str = None)`
Plot 2D slices of a 3D property.

**Parameters:**
- `data` (dict): Grid data dictionary
- `property_name` (str): Property to plot
- `save_path` (str): Optional path to save figure

##### `plot_model_comparison(results: dict, save_path: str = None)`
Plot comparison of multiple model results.

**Parameters:**
- `results` (dict): Dictionary of model results
- `save_path` (str): Optional path to save figure

##### `plot_uncertainty_map(predictions, uncertainties, save_path: str = None)`
Plot prediction uncertainty visualization.

**Parameters:**
- `predictions` (array): Model predictions
- `uncertainties` (array): Prediction uncertainties
- `save_path` (str): Optional path to save figure

---

### Model Classes

#### `SPE9GPModel`
Standard Gaussian Process model for spatial data.

```python
from spe9_geomodeling import SPE9GPModel

model = SPE9GPModel(kernel_type='rbf', ard=True, input_dim=3)
```

**Parameters:**
- `kernel_type` (str): Kernel type ('rbf', 'matern', 'spectral_mixture')
- `ard` (bool): Use Automatic Relevance Determination
- `input_dim` (int): Input dimensionality

#### `DeepGPModel`
Deep Gaussian Process model with neural network features.

```python
from spe9_geomodeling import DeepGPModel

model = DeepGPModel(hidden_dims=[64, 32], kernel_type='rbf', input_dim=3)
```

**Parameters:**
- `hidden_dims` (list): Hidden layer dimensions
- `kernel_type` (str): Kernel type for final GP layer
- `input_dim` (int): Input dimensionality

---

### Utility Functions

#### `load_spe9_data(file_path: str = None)`
Convenience function to load SPE9 dataset.

**Parameters:**
- `file_path` (str): Optional path to SPE9.GRDECL file

**Returns:**
- `dict`: Loaded SPE9 data

#### `create_gp_model(model_type: str, **kwargs)`
Factory function for creating GP models.

**Parameters:**
- `model_type` (str): Type of model ('standard', 'deep', 'sparse')
- `**kwargs`: Model-specific parameters

**Returns:**
- GP model instance

---

## 🧪 Experiments Package: `spe9_geomodeling.experiments`

### `DeepGPExperiment`
Comprehensive experiment framework for comparing GP models.

```python
from spe9_geomodeling import DeepGPExperiment

experiment = DeepGPExperiment()
```

**Methods:**

##### `run_comparison_experiment(train_samples: int = 500, test_samples: int = 100, training_iterations: int = 50)`
Run comprehensive model comparison.

**Parameters:**
- `train_samples` (int): Number of training samples
- `test_samples` (int): Number of test samples  
- `training_iterations` (int): Training iterations per model

**Returns:**
- `dict`: Results dictionary with metrics for each model

##### `load_data(file_path: str = None)`
Load data for experiments.

**Parameters:**
- `file_path` (str): Optional path to data file

##### `train_model(model, likelihood, X_train, y_train, training_iter: int = 50)`
Train a single model.

**Parameters:**
- `model`: Model to train
- `likelihood`: Model likelihood
- `X_train`: Training features
- `y_train`: Training targets
- `training_iter` (int): Number of training iterations

**Returns:**
- `tuple`: (trained_model, training_time)

##### `evaluate_model(model, likelihood, X_test, y_test)`
Evaluate a trained model.

**Parameters:**
- `model`: Trained model
- `likelihood`: Model likelihood
- `X_test`: Test features
- `y_test`: Test targets

**Returns:**
- `dict`: Evaluation metrics

##### `plot_comparison_results(results: dict, save_path: str = 'deep_gp_comparison.png')`
Plot experiment results.

**Parameters:**
- `results` (dict): Experiment results
- `save_path` (str): Path to save figure

---

## 📊 Data Structures

### EvaluationResults
Result object returned by model evaluation.

**Attributes:**
- `r2` (float): R-squared score
- `rmse` (float): Root Mean Square Error
- `mae` (float): Mean Absolute Error
- `predictions` (array): Model predictions
- `residuals` (array): Prediction residuals

### ExperimentResults
Result dictionary structure for experiments.

**Structure:**
```python
{
    'model_name': {
        'metrics': {
            'r2_score': float,
            'rmse': float,
            'mae': float
        },
        'training_time': float,
        'predictions': array,
        'uncertainties': array
    }
}
```

---

## 🔧 Configuration

### Environment Variables
- `SPE9_DATA_PATH`: Default path to SPE9.GRDECL file
- `SPE9_CACHE_DIR`: Directory for caching processed data
- `SPE9_DEVICE`: Device for PyTorch computations ('cpu', 'cuda')

### Default Parameters
```python
DEFAULT_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'training_iterations': 50,
    'inducing_points': 100,
    'learning_rate': 0.1,
    'kernel_lengthscale': 1.0,
    'noise_variance': 1e-4
}
```

---

## 🐛 Exception Handling

### Custom Exceptions

#### `SPE9DataError`
Raised when there are issues with data loading or processing.

#### `ModelTrainingError`
Raised when model training fails.

#### `EvaluationError`
Raised when model evaluation encounters errors.

### Error Handling Example
```python
from spe9_geomodeling import UnifiedSPE9Toolkit, SPE9DataError

try:
    toolkit = UnifiedSPE9Toolkit()
    data = load_spe9_data()
    toolkit.load_spe9_data(data)
except SPE9DataError as e:
    print(f"Data loading failed: {e}")
    # Fallback to synthetic data
    toolkit.load_synthetic_data()
```

---

## 📝 Type Hints

The toolkit provides comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator

def create_train_test_split(
    self, 
    test_size: float = 0.2,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

---

**For more examples and advanced usage, see the [Examples](examples.md) and [Deep GP Guide](deep_gp.md) documentation.**
