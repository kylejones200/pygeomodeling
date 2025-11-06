# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-11-05

### Changed
- Repository cleanup - removed temporary deployment files
- Added proper CHANGELOG.md for version tracking

## [0.2.0] - 2025-11-05

### Added

#### Core Features
- **Error Handling**: Custom exception classes with descriptive messages and suggestions
  - `PyGeoModelingError` base class and 10+ specific exception types
  - Input validation throughout codebase
  - Helper functions for common error scenarios
- **Model Serialization**: Save and load models with full metadata
  - `ModelMetadata` class for version tracking
  - `ModelSerializer` for persistence operations
  - Support for joblib, pickle, and torch formats
- **Spatial Cross-Validation**: Proper validation for geostatistical data
  - `SpatialKFold` - spatial block-based K-fold CV
  - `BlockCV` - block cross-validation with buffer zones
  - `cross_validate_spatial()` function
- **Hyperparameter Tuning**: Optuna integration for Bayesian optimization
  - `HyperparameterTuner` class
  - Automatic best model selection
- **Parallel Processing**: Multi-core training and prediction
  - `ParallelModelTrainer` - train multiple models simultaneously
  - `BatchPredictor` - batch predictions for large datasets
  - `ParallelCrossValidator` - parallel fold evaluation
  - 3-4x speedup on multi-core systems

#### CI/CD & Tooling
- GitHub Actions workflow for multi-OS/Python testing
- Pre-commit hooks for code quality enforcement
- Automated development setup script (`setup_dev.sh`)
- Build and publish automation scripts

#### Documentation
- `CONTRIBUTING.md` - Comprehensive contribution guidelines
- `QUICK_START.md` - 5-minute quick start guide
- `ADVANCED_FEATURES.md` - Complete feature documentation
- Tutorial Jupyter notebooks:
  - `01_getting_started.ipynb` - Beginner tutorial
  - `02_advanced_modeling.ipynb` - Advanced features
- API documentation for all new modules

#### Sample Data
- `data/sample_small.grdecl` - Small 5×5×3 sample grid for testing
- Data documentation and usage examples

#### Examples
- `examples/advanced_workflow.py` - Complete workflow demonstration
- Interactive notebooks with visualizations

### Changed
- Updated `grdecl_parser.py` with comprehensive input validation
- Enhanced `__init__.py` to export new modules
- Improved error messages throughout codebase
- Updated README with new features and badges
- Version bump from 0.1.2 to 0.2.0

### Performance
- 3.75x speedup for parallel model training (8 cores)
- 3.6x speedup for parallel cross-validation (8 cores)
- 3.1x speedup for batch predictions (8 cores)

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- Added bandit security scanning in pre-commit hooks
- Added security checks in CI/CD workflow

## [0.1.2] - Previous Release

### Added
- Initial public release
- GRDECL parser for Eclipse files
- Unified toolkit for sklearn and GPyTorch
- Basic visualization utilities
- Traditional and Deep GP models

---

## Version History

- **0.2.1** (2025-11-05) - Repository cleanup
- **0.2.0** (2025-11-05) - Production-ready features
- **0.1.2** (Previous) - Initial release

[0.2.1]: https://github.com/kylejones200/pygeomodeling/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kylejones200/pygeomodeling/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/kylejones200/pygeomodeling/releases/tag/v0.1.2
