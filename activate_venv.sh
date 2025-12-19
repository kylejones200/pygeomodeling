#!/bin/bash
# Activation script for SPE9 Geomodeling virtual environment

echo "🚀 Activating SPE9 Geomodeling virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python location: $(which python)"
echo "📋 Installed packages:"
echo "   - pygeomodeling (development mode)"
echo "   - All core dependencies (numpy, pandas, scikit-learn, etc.)"
echo "   - Development tools (black, pytest, jupyter)"
echo "   - Documentation tools (mkdocs, mkdocs-material)"
echo ""
echo "🔧 Available commands:"
echo "   - pytest                    # Run tests"
echo "   - black .                   # Format code"
echo "   - mkdocs serve              # Serve docs locally"
echo "   - mkdocs build              # Build docs"
echo "   - jupyter lab               # Start Jupyter Lab"
echo ""
echo "💡 To deactivate: deactivate"
