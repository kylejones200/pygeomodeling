# Installation Guide

This guide covers all installation methods for the SPE9 Geomodeling Toolkit.

## 📦 Installation Methods

### Method 1: PyPI Installation (Recommended)

#### Basic Installation

For traditional Gaussian Process models only:

```bash
pip install spe9-geomodeling
```

#### Advanced Installation

For Deep GP models with GPyTorch support:

```bash
pip install spe9-geomodeling[advanced]
```

#### Complete Installation

For all features including geospatial tools:

```bash
pip install spe9-geomodeling[all]
```

### Method 2: Development Installation

For contributors and developers:

```bash
git clone https://github.com/yourusername/spe9-geomodeling.git
cd spe9-geomodeling
pip install -e ".[dev]"
```

### Method 3: From Source

```bash
git clone https://github.com/yourusername/spe9-geomodeling.git
cd spe9-geomodeling
pip install .
```

## 🔧 Optional Dependencies

The toolkit uses modular dependencies to keep the base installation lightweight:

### Core Dependencies (Always Installed)

- `numpy >= 1.24.0` - Numerical computing
- `pandas >= 1.5.0` - Data manipulation
- `scikit-learn >= 1.3.0` - Machine learning
- `matplotlib >= 3.7.0` - Basic plotting
- `pykrige >= 1.6.0` - Kriging algorithms

### Advanced Dependencies

Install with `pip install spe9-geomodeling[advanced]`:

- `torch >= 2.0.0` - Deep learning framework
- `gpytorch >= 1.11.0` - Gaussian Process library
- `botorch >= 0.9.0` - Bayesian optimization
- `optuna >= 3.3.0` - Hyperparameter optimization

### Geospatial Dependencies

Install with `pip install spe9-geomodeling[geospatial]`:

- `rasterio >= 1.3.0` - Raster data I/O
- `geopandas >= 0.13.0` - Geospatial data analysis
- `shapely >= 2.0.0` - Geometric operations
- `xarray >= 2023.1.0` - N-dimensional arrays

### Visualization Dependencies

Install with `pip install spe9-geomodeling[visualization]`:

- `plotly >= 5.15.0` - Interactive plots
- `seaborn >= 0.12.0` - Statistical visualization
- `ipywidgets >= 8.0.0` - Jupyter widgets

### Development Dependencies

Install with `pip install spe9-geomodeling[dev]`:

- `pytest >= 7.4.0` - Testing framework
- `black >= 23.0.0` - Code formatting
- `jupyter >= 1.0.0` - Notebook environment

## 🖥️ System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB
- **Storage**: 1GB free space
- **OS**: Windows 10, macOS 10.15, or Linux

### Recommended Requirements

- **Python**: 3.11 or higher
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **GPU**: CUDA-compatible GPU for Deep GP models (optional)

## 🔍 Verification

After installation, verify everything works:

```python
# Test basic installation
import spe9_geomodeling
print(f"SPE9 Toolkit version: {spe9_geomodeling.__version__}")

# Test core functionality
from spe9_geomodeling import UnifiedSPE9Toolkit
toolkit = UnifiedSPE9Toolkit()
print("✅ Core functionality available")

# Test advanced features (if installed)
try:
    from spe9_geomodeling import DeepGPExperiment
    print("✅ Advanced Deep GP features available")
except ImportError:
    print("ℹ️ Advanced features not installed (install with [advanced])")
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: ImportError for GPyTorch

```
ImportError: No module named 'gpytorch'
```

**Solution**: Install advanced dependencies:

```bash
pip install spe9-geomodeling[advanced]
```

#### Issue: CUDA/GPU Issues

```
RuntimeError: CUDA out of memory
```

**Solution**: Use CPU-only mode or reduce batch size:

```python
import torch
torch.cuda.is_available()  # Check if CUDA is available
# Force CPU usage if needed
device = torch.device('cpu')
```

#### Issue: File Not Found for SPE9 Data

```
FileNotFoundError: SPE9.GRDECL not found
```

**Solution**: Ensure SPE9 dataset is available or provide correct path:

```python
from spe9_geomodeling import GRDECLParser
parser = GRDECLParser('/path/to/your/SPE9.GRDECL')
```

### Environment-Specific Issues

#### Conda Environments

If using conda, create a dedicated environment:

```bash
conda create -n spe9 python=3.11
conda activate spe9
pip install spe9-geomodeling[all]
```

#### Apple Silicon (M1/M2) Macs

For optimal performance:

```bash
# Install with Apple Silicon optimized PyTorch
pip install spe9-geomodeling[advanced] --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Windows with CUDA

For GPU acceleration on Windows:

```bash
# Install CUDA-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install spe9-geomodeling[advanced]
```

## 📊 Performance Testing

Test your installation performance:

```python
from spe9_geomodeling import DeepGPExperiment
import time

# Quick performance test
start_time = time.time()
experiment = DeepGPExperiment()
# This will use synthetic data if SPE9 is not available
results = experiment.run_comparison_experiment()
elapsed = time.time() - start_time

print(f"Performance test completed in {elapsed:.1f} seconds")
print(f"Best model R²: {max(r['metrics']['r2_score'] for r in results.values()):.3f}")
```

## 🔄 Updating

To update to the latest version:

```bash
pip install --upgrade spe9-geomodeling
```

To update with all dependencies:

```bash
pip install --upgrade spe9-geomodeling[all]
```

## 🗑️ Uninstallation

To completely remove the toolkit:

```bash
pip uninstall spe9-geomodeling
```

---

**Next Steps**: Once installed, check out the [Quick Start Tutorial](quickstart.md) to begin using the toolkit!
