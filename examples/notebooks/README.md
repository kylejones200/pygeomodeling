# PyGeomodeling Tutorial Notebooks

Interactive Jupyter notebooks for learning PyGeomodeling.

## Notebooks

### 1. Getting Started (`01_getting_started.ipynb`)
**Level**: Beginner  
**Time**: 15-20 minutes

Learn the basics:
- Loading GRDECL files
- Exploring reservoir properties
- Preparing features
- Training your first model
- Making predictions

**Prerequisites**: Basic Python knowledge

### 2. Advanced Modeling (`02_advanced_modeling.ipynb`)
**Level**: Intermediate  
**Time**: 20-30 minutes

Advanced features:
- Spatial cross-validation
- Parallel model training
- Model serialization
- Model comparison

**Prerequisites**: Complete notebook 01

### 3. Variogram Analysis (`03_variogram_analysis.ipynb`)
**Level**: Intermediate  
**Time**: 25-35 minutes

Spatial correlation analysis:
- Experimental variogram computation
- Model fitting (spherical, exponential, Gaussian)
- Anisotropy detection
- Interpretation and validation

**Prerequisites**: Complete notebook 01

### 4. Ore Grade Forecasting (`04_ore_grade_forecasting.ipynb`)
**Level**: Advanced  
**Time**: 35-45 minutes

Real-world mining application:
- Geochemical data integration
- Pathfinder element analysis
- Uncertainty quantification for drilling
- Model comparison (GPR vs XGBoost)

**Prerequisites**: Complete notebooks 01-03

### 5. Hyperparameter Tuning (Coming Soon)
**Level**: Advanced  
**Time**: 30-40 minutes

Optimize your models:
- Optuna integration
- Bayesian optimization
- Search space design
- Parallel tuning

### 6. Production Deployment (Coming Soon)
**Level**: Advanced  
**Time**: 40-50 minutes

Deploy models:
- Model versioning
- API creation
- Monitoring
- Best practices

## Running the Notebooks

### Option 1: Local Jupyter

```bash
# Install Jupyter
pip install jupyter

# Navigate to notebooks directory
cd examples/notebooks

# Start Jupyter
jupyter notebook
```

### Option 2: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

### Option 3: VS Code

1. Install the Jupyter extension
2. Open a `.ipynb` file
3. Select Python kernel
4. Run cells

### Option 4: Google Colab

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Install pygeomodeling:
   ```python
   !pip install pygeomodeling
   ```

## Data Files

Notebooks use sample data from `../../data/`:
- `sample_small.grdecl` - Small 5×5×3 grid for quick testing
- `SPE9.GRDECL` - Full SPE9 dataset (if available)

## Tips

- **Run cells in order** - Each cell may depend on previous ones
- **Restart kernel** if you encounter issues
- **Save your work** regularly
- **Experiment** - Modify parameters and see what happens!

## Troubleshooting

### Import Errors

```bash
# Install in development mode
cd ../..
pip install -e ".[all]"
```

### Kernel Issues

```bash
# Install ipykernel
pip install ipykernel

# Add kernel
python -m ipykernel install --user --name=pygeomodeling
```

### Data Not Found

Check that you're running from the correct directory:
```python
import os
print(os.getcwd())
```

## Contributing

Have ideas for new tutorials? Please:
1. Open an issue describing the tutorial
2. Submit a PR with your notebook
3. Follow the existing format

## Questions?

- GitHub Issues: [Report issues](https://github.com/kylejones200/pygeomodeling/issues)
- Email: kyletjones@gmail.com
- Documentation: [Full docs](../../docs/)
