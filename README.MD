# PyGeomodeling

[![PyPI version](https://badge.fury.io/py/pygeomodeling.svg)](https://badge.fury.io/py/pygeomodeling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kylejones200/pygeomodeling/workflows/Tests/badge.svg)](https://github.com/kylejones200/pygeomodeling/actions)
[![Documentation](https://readthedocs.org/projects/pygeomodeling/badge/?version=latest)](https://pygeomodeling.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Advanced Gaussian Process Regression and Kriging toolkit for SPE9 reservoir modeling. This toolkit provides a comprehensive framework for spatial modeling of reservoir properties using both traditional Gaussian Process models and cutting-edge Deep Gaussian Process architectures.

## Key Features

### Core Capabilities

- **GRDECL Parser**: Load and parse Eclipse GRDECL files with automatic property extraction
- **Unified Toolkit**: Single interface supporting both scikit-learn and GPyTorch workflows
- **Advanced Models**: Traditional GP (RBF, Matérn, Combined kernels) and Deep GP with neural network features
- **Rich Visualization**: Comprehensive plotting utilities for model comparison and spatial analysis

### Features (v0.3.0)

- **Error Handling**: Comprehensive validation with helpful error messages
- **Model Serialization**: Save/load models with versioning and metadata
- **Spatial Cross-Validation**: Proper validation for geostatistical data
- **Parallel Processing**: 3-4x speedup with multi-core training
- **Hyperparameter Tuning**: Optuna integration for optimization
- **Variogram Analysis**: Semi-variogram modeling for spatial correlation (NEW!)
- **Tutorial Notebooks**: Interactive learning with Jupyter
- **Research-Ready**: Built for reproducible scientific research

## Installation

### Basic Installation (Traditional GP Models)

```bash
pip install pygeomodeling
```

### With Advanced Deep GP Features

```bash
pip install pygeomodeling[advanced]
```

### Complete Installation (All Features)

```bash
pip install pygeomodeling[all]
```

### Development Installation

```bash
git clone https://github.com/kylejones200/pygeomodeling.git
cd pygeomodeling
./setup_dev.sh  # Automated setup
# or manually: pip install -e ".[dev,docs,all]"
```

## Quick Start

### Basic Usage

```python
from spe9_geomodeling import GRDECLParser, UnifiedSPE9Toolkit

# Load SPE9 dataset
parser = GRDECLParser('path/to/SPE9.GRDECL')
data = parser.load_data()

# Create toolkit and train model
toolkit = UnifiedSPE9Toolkit()
toolkit.load_spe9_data(data)
X_train, X_test, y_train, y_test = toolkit.create_train_test_split()

# Train traditional GP model
model = toolkit.create_sklearn_model('gpr', kernel_type='combined')
toolkit.train_sklearn_model(model, 'combined_gpr')
results = toolkit.evaluate_model('combined_gpr', X_test, y_test)

print(f"R² Score: {results.r2:.4f}")
```

### Deep GP Experiment

```python
from spe9_geomodeling import DeepGPExperiment

# Run comprehensive comparison
experiment = DeepGPExperiment()
results = experiment.run_comparison_experiment()

# Results automatically saved to deep_gp_comparison.png
print(f"Best model: {max(results.keys(), key=lambda x: results[x]['metrics']['r2_score'])}")
```

## What's Included

### Core Modules

- **`grdecl_parser.py`**: Eclipse GRDECL file parser with SPE9 support
- **`unified_toolkit.py`**: Main toolkit with sklearn and GPyTorch backends
- **`model_gp.py`**: Advanced GPyTorch models (Standard GP, Deep GP)
- **`plot.py`**: Comprehensive visualization utilities
- **`deep_gp_experiment.py`**: Ready-to-run comparison experiments

### Example Scripts

- **`main.py`**: Traditional geomodeling workflow
- **`package.py`**: Packaging and distribution helper

## Scientific Applications

This toolkit has been designed for:

- **Reservoir Modeling**: Spatial interpolation of permeability and porosity
- **Uncertainty Quantification**: Gaussian Process uncertainty estimates
- **Model Comparison**: Traditional vs Deep GP performance analysis
- **Geostatistical Research**: Advanced spatial modeling techniques

## Model Performance

Based on SPE9 reservoir dataset analysis:

- **Traditional GP (Combined Kernel)**: R² = 0.277 (Best)
- **Deep GP (Small)**: R² = 0.189
- **Training Time**: ~1.3-1.8 seconds per model

*Traditional Gaussian Process models with combined RBF+Matérn kernels outperform Deep GP models for SPE9 spatial patterns, demonstrating the effectiveness of classical geostatistical approaches for this dataset.*

## Dependencies

### Core Dependencies (Always Installed)

- numpy >= 1.24.0
- pandas >= 1.5.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- pykrige >= 1.6.0

### Optional Dependencies

- **Advanced**: torch, gpytorch, botorch, optuna (for Deep GP models)
- **Geospatial**: rasterio, geopandas, shapely (for advanced spatial analysis)
- **Performance**: numba (for JIT compilation)
- **Visualization**: plotly, ipywidgets (for interactive plots)

## Documentation

Detailed documentation and examples are available in the repository:

- [API Reference](docs/api.md)
- [Tutorial Notebooks](examples/)
- [Model Comparison Guide](docs/model_comparison.md)

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/spe9-geomodeling/issues)
- **Email**: <kyletjones@gmail.com>
- **Documentation**: [Project Wiki](https://github.com/yourusername/spe9-geomodeling/wiki)

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{jones2025spe9geomodeling,
  title={SPE9 Geomodeling Toolkit: Advanced Gaussian Process Regression for Reservoir Modeling},
  author={Jones, Kyle T.},
  year={2025},
  url={https://github.com/kylejones200/pygeomodeling}
}
```

## Outputs

- `PERMX_GPR.GRDECL`: predicted permeability in GRDECL format
- `SIGMA_GPR.GRDECL`: standard deviation (uncertainty) of predictions
- `gpr_prediction_slices.png`: side-by-side slice visualization
