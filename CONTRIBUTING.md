# Contributing to PyGeomodeling

Thank you for your interest in contributing to PyGeomodeling! This document provides guidelines and instructions for contributing.

## Ways to Contribute

- Report bugs and suggest features via GitHub Issues
- Improve documentation - fix typos, add examples, clarify explanations
- Submit bug fixes - help resolve existing issues
- Add new features - implement new geomodeling algorithms or utilities
- Write tests - improve test coverage
- Share examples - contribute tutorial notebooks or example workflows

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/pygeomodeling.git
cd pygeomodeling
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create a new branch for your feature or fix
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
# Format code
black spe9_geomodeling tests

# Check linting
flake8 spe9_geomodeling tests
```

Key conventions:

- Line length: 88 characters (Black default)
- Use type hints where possible
- Follow PEP 8 naming conventions
- Write docstrings for all public functions/classes (Google style)

### Writing Tests

All new features and bug fixes should include tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spe9_geomodeling --cov-report=html
```

Test guidelines:

- Place tests in tests/ directory
- Name test files test_*.py
- Name test functions test_*
- Use descriptive test names
- Mock external dependencies
- Aim for >80% code coverage

### Documentation

Update documentation for any user-facing changes:

```bash
# Build documentation locally
mkdocs serve
```

### Commit Messages

Write clear, descriptive commit messages:

```
feat: add spatial cross-validation with buffer zones

- Implement BlockCV class with configurable buffer sizes
- Add tests for edge cases
- Update documentation with examples

Closes #123
```

Commit message format:

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Test additions/changes
- refactor: Code refactoring
- perf: Performance improvements
- chore: Maintenance tasks

## Testing Checklist

Before submitting a PR, ensure:

- All tests pass: pytest tests/
- Code is formatted: black spe9_geomodeling tests
- Linting passes: flake8 spe9_geomodeling tests
- Documentation builds: mkdocs build
- New features have tests
- New features have documentation

## Submitting a Pull Request

1. Push your changes
2. Create a Pull Request on GitHub
3. Use a clear, descriptive title
4. Reference related issues (e.g., "Fixes #123")
5. Describe what changed and why
6. Respond to reviewer feedback
7. Once approved, a maintainer will merge your PR

## Project Structure

```
pygeomodeling/
├── spe9_geomodeling/       # Main package
│   ├── grdecl_parser.py    # GRDECL file parsing
│   ├── unified_toolkit.py  # Main toolkit
│   ├── model_gp.py         # GP models
│   ├── plot.py             # Visualization
│   ├── exceptions.py       # Custom exceptions
│   ├── serialization.py    # Model persistence
│   ├── cross_validation.py # Spatial CV
│   └── parallel.py         # Parallel processing
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Example scripts
└── data/                   # Sample data
```

## Questions?

- Open an issue for questions
- Email: <kyletjones@gmail.com>
- Check existing issues and PRs first

Thank you for contributing!
