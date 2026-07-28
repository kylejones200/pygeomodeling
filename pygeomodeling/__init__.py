"""
PyGeomodeling Toolkit.

Advanced Gaussian Process Regression and Kriging toolkit for reservoir modeling.
Supports both traditional GP models and Deep GP models for spatial pattern analysis.
"""

__version__ = "0.3.2"
__author__ = "K. Jones"
__email__ = "kyletjones@gmail.com"

# Import main classes for easy access
try:
    from .grdecl_parser import GRDECLParser, load_spe9_data
    from .plot import SPE9Plotter

    # Deprecated: Import old toolkit for backward compatibility
    from .toolkit import SPE9Toolkit as _SPE9Toolkit
    from .unified_toolkit import UnifiedSPE9Toolkit

    SPE9Toolkit = _SPE9Toolkit  # Keep for backward compatibility
except ImportError:
    # Handle case where optional dependencies aren't installed
    pass

# Import model classes if GPyTorch is available
try:
    from .model_gp import (
        BayesianNeuralNetwork,
        DeepGPModel,
        MultiOutputGPModel,
        NeuralNetworkGPModel,
        SPE9GPModel,
        SparseGPModel,
        create_gp_model,
    )
    from .advanced_kernels import (
        NonStationaryRBFKernel,
        SpectralMixtureKernel,
        create_advanced_kernel,
    )
except (
    ImportError
) as exc:  # pragma: no cover - executed only when optional deps missing
    # Store exception for use in closures
    _gp_import_error = exc

    class _MissingGPDependency:
        """Placeholder that raises a helpful error at runtime."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "Optional dependency for Gaussian Process models is missing. "
                "Install the 'advanced' extras (pip install pygeomodeling[advanced]) "
                "to enable SPE9GPModel support."
            ) from _gp_import_error

    class SPE9GPModel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder SPE9 Gaussian Process model."""

    class DeepGPModel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Deep Gaussian Process model."""

    class MultiOutputGPModel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Multi-Output Gaussian Process model."""

    class SparseGPModel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Sparse Gaussian Process model."""

    class NeuralNetworkGPModel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Neural Network Gaussian Process model."""

    class BayesianNeuralNetwork(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Bayesian Neural Network model."""

    class NonStationaryRBFKernel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Non-Stationary RBF Kernel."""

    class SpectralMixtureKernel(_MissingGPDependency):  # type: ignore[no-redef]
        """Placeholder Spectral Mixture Kernel."""

    def create_gp_model(*args, **kwargs):  # type: ignore[no-redef,misc]
        raise ImportError(
            "Optional dependency for Gaussian Process models is missing. "
            "Install the 'advanced' extras (pip install pygeomodeling[advanced]) "
            "to enable GP model support."
        ) from _gp_import_error

    def create_advanced_kernel(*args, **kwargs):  # type: ignore[no-redef,misc]
        raise ImportError(
            "Optional dependency for advanced kernels is missing. "
            "Install the 'advanced' extras (pip install pygeomodeling[advanced]) "
            "to enable advanced kernel support."
        ) from _gp_import_error


# Import experimental modules
try:
    from .experiments import DeepGPExperiment
except ImportError:
    # Experimental modules not available
    pass

# Import new advanced features
try:
    from . import exceptions
    from .confidence_scoring import (
        ConfidenceScore,
        ConfidenceScorer,
        WellConfidenceReport,
        compare_confidence_across_wells,
        export_review_list,
    )
    from .cross_validation import (
        BlockCV,
        HyperparameterTuner,
        SpatialKFold,
        cross_validate_spatial,
    )
    from .facies import (
        FACIES_LABELS,
        FaciesClassificationResult,
        FaciesClassifier,
        load_facies_data,
        prepare_facies_features,
    )
    from .formation_tops import (
        BoundaryDetectionResult,
        FormationTop,
        FormationTopDetector,
        compare_tops_with_reference,
    )
    from .integration_exports import (
        FaciesLogExporter,
        FormationTopExporter,
        LASExporter,
        PetrelProjectExporter,
        create_correction_template,
        import_expert_corrections,
    )
    from .kriging import (
        CoKriging,
        KrigingResult,
        OrdinaryKriging,
        UniversalKriging,
        simple_kriging,
    )
    from .log_features import FeatureSet, LogFeatureEngineer, prepare_ml_dataset
    from .parallel import (
        BatchPredictor,
        ParallelCrossValidator,
        ParallelModelTrainer,
        parallel_grid_search,
    )
    from .reservoir_engineering import (
        PetrophysicsCalculator,
        ReservoirType,
        VolumetricResult,
        VolumetricsCalculator,
        calculate_reserves_uncertainty,
        decline_curve_analysis,
    )
    from .serialization import ModelMetadata, ModelSerializer, load_model, save_model
    from .variogram import (
        VariogramModel,
        compute_experimental_variogram,
        cross_validation_variogram,
        directional_variogram,
        fit_variogram_model,
        predict_variogram,
    )
    from .variogram_plot import (
        plot_directional_variograms,
        plot_variogram,
        plot_variogram_cloud,
        plot_variogram_comparison,
    )
    from .well_data import (
        CurveInfo,
        LASParser,
        WellHeader,
        WellLogUpscaler,
        load_las_file,
        upscale_well_logs,
    )
    from .well_log_processor import (
        CURVE_SIGNATURES,
        CurveQuality,
        ProcessedWellLogs,
        WellLogProcessor,
        process_multiple_wells,
    )
    from .workflow_manager import (
        CorrectionRecord,
        WorkflowIteration,
        WorkflowManager,
        WorkflowState,
        create_workflow_dashboard,
    )
    from .reservoir_formats import (
        ASCIIGridReader,
        GridMetadata,
        PetrelASCIIReader,
        PetrelBinaryReader,
        RESQMLReader,
        ReservoirGrid,
        load_reservoir_data,
    )
    from .spatial_reference import (
        SpatialReference,
        get_crs_from_epsg,
        transform_coordinates,
    )
except ImportError:
    # Advanced features not available
    pass

__all__ = [
    # Core modules
    "GRDECLParser",
    "load_spe9_data",
    "UnifiedSPE9Toolkit",
    "SPE9Toolkit",  # Deprecated, use UnifiedSPE9Toolkit
    "SPE9Plotter",
    # Model classes
    "SPE9GPModel",
    "DeepGPModel",
    "MultiOutputGPModel",
    "SparseGPModel",
    "NeuralNetworkGPModel",
    "BayesianNeuralNetwork",
    "create_gp_model",
    # Advanced kernels
    "NonStationaryRBFKernel",
    "SpectralMixtureKernel",
    "create_advanced_kernel",
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
    # Variogram analysis
    "VariogramModel",
    "compute_experimental_variogram",
    "fit_variogram_model",
    "predict_variogram",
    "directional_variogram",
    "cross_validation_variogram",
    "plot_variogram",
    "plot_variogram_comparison",
    "plot_directional_variograms",
    "plot_variogram_cloud",
    # Kriging
    "OrdinaryKriging",
    "UniversalKriging",
    "CoKriging",
    "simple_kriging",
    "KrigingResult",
    # Well data
    "LASParser",
    "WellHeader",
    "CurveInfo",
    "WellLogUpscaler",
    "load_las_file",
    "upscale_well_logs",
    # Reservoir engineering
    "VolumetricsCalculator",
    "PetrophysicsCalculator",
    "VolumetricResult",
    "ReservoirType",
    "calculate_reserves_uncertainty",
    "decline_curve_analysis",
    # Facies classification
    "FaciesClassifier",
    "FaciesClassificationResult",
    "FACIES_LABELS",
    "load_facies_data",
    "prepare_facies_features",
    # Well log processing
    "WellLogProcessor",
    "ProcessedWellLogs",
    "CurveQuality",
    "CURVE_SIGNATURES",
    "process_multiple_wells",
    # Log feature engineering
    "LogFeatureEngineer",
    "FeatureSet",
    "prepare_ml_dataset",
    # Formation tops
    "FormationTopDetector",
    "FormationTop",
    "BoundaryDetectionResult",
    "compare_tops_with_reference",
    # Confidence scoring
    "ConfidenceScorer",
    "ConfidenceScore",
    "WellConfidenceReport",
    "compare_confidence_across_wells",
    "export_review_list",
    # Integration & exports
    "LASExporter",
    "FormationTopExporter",
    "FaciesLogExporter",
    "PetrelProjectExporter",
    "create_correction_template",
    "import_expert_corrections",
    # Workflow management
    "WorkflowManager",
    "WorkflowIteration",
    "CorrectionRecord",
    "WorkflowState",
    "create_workflow_dashboard",
    # Reservoir formats
    "load_reservoir_data",
    "ReservoirGrid",
    "GridMetadata",
    "PetrelASCIIReader",
    "PetrelBinaryReader",
    "RESQMLReader",
    "ASCIIGridReader",
    # Spatial reference
    "SpatialReference",
    "transform_coordinates",
    "get_crs_from_epsg",
    # Exceptions
    "exceptions",
]
