"""
SPE9 Geomodeling Toolkit

Advanced Gaussian Process Regression and Kriging toolkit for SPE9 reservoir modeling.
Supports both traditional GP models and Deep GP models for spatial pattern analysis.
"""

__version__ = "0.1.0"
__author__ = "K. Jones"
__email__ = "kyletjones@gmail.com"

# Import main classes for easy access
try:
    from .grdecl_parser import GRDECLParser, load_spe9_data
    from .unified_toolkit import UnifiedSPE9Toolkit
    from .plot import SPE9Plotter
    from .spe9_toolkit import SPE9Toolkit
except ImportError:
    # Handle case where optional dependencies aren't installed
    pass

# Import model classes if GPyTorch is available
try:
    from .model_gp import SPE9GPModel, DeepGPModel, create_gp_model
except ImportError:
    # GPyTorch not available
    pass

# Import experimental modules
try:
    from .experiments import DeepGPExperiment
except ImportError:
    # Experimental modules not available
    pass

__all__ = [
    "GRDECLParser",
    "load_spe9_data",
    "UnifiedSPE9Toolkit",
    "SPE9Toolkit",
    "SPE9Plotter",
    "SPE9GPModel",
    "DeepGPModel",
    "create_gp_model",
    "DeepGPExperiment",
]
