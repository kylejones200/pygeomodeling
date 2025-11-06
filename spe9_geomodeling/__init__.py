"""
PyGeomodeling Toolkit

Advanced Gaussian Process Regression and Kriging toolkit for reservoir modeling.
Supports both traditional GP models and Deep GP models for spatial pattern analysis.
"""

__version__ = "0.1.0"
__author__ = "K. Jones"
__email__ = "kyletjones@gmail.com"

# Import main classes for easy access
try:
    from .grdecl_parser import GRDECLParser, load_spe9_data
    from .toolkit import SPE9Toolkit
    from .plot import SPE9Plotter
    from .unified_toolkit import UnifiedSPE9Toolkit
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

# Import new advanced features
try:
    from . import exceptions
    from .serialization import (
        ModelMetadata,
        ModelSerializer,
        save_model,
        load_model,
    )
    from .cross_validation import (
        SpatialKFold,
        BlockCV,
        cross_validate_spatial,
        HyperparameterTuner,
    )
    from .parallel import (
        ParallelModelTrainer,
        BatchPredictor,
        ParallelCrossValidator,
        parallel_grid_search,
    )
except ImportError:
    # Advanced features not available
    pass

__all__ = [
    # Core modules
    "GRDECLParser",
    "load_spe9_data",
    "SPE9Toolkit",
    "UnifiedSPE9Toolkit",
    "SPE9Plotter",
    # Model classes
    "SPE9GPModel",
    "DeepGPModel",
    "create_gp_model",
    # Experiments
    "DeepGPExperiment",
    # Serialization
    "ModelMetadata",
    "ModelSerializer",
    "save_model",
    "load_model",
    # Cross-validation
    "SpatialKFold",
    "BlockCV",
    "cross_validate_spatial",
    "HyperparameterTuner",
    # Parallel processing
    "ParallelModelTrainer",
    "BatchPredictor",
    "ParallelCrossValidator",
    "parallel_grid_search",
    # Exceptions
    "exceptions",
]
