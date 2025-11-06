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
    from .variogram import (
        VariogramModel,
        compute_experimental_variogram,
        fit_variogram_model,
        predict_variogram,
        directional_variogram,
        cross_validation_variogram,
    )
    from .variogram_plot import (
        plot_variogram,
        plot_variogram_comparison,
        plot_directional_variograms,
        plot_variogram_cloud,
    )
    from .kriging import (
        OrdinaryKriging,
        UniversalKriging,
        CoKriging,
        simple_kriging,
        KrigingResult,
    )
    from .well_data import (
        LASParser,
        WellHeader,
        CurveInfo,
        WellLogUpscaler,
        load_las_file,
        upscale_well_logs,
    )
    from .reservoir_engineering import (
        VolumetricsCalculator,
        PetrophysicsCalculator,
        VolumetricResult,
        ReservoirType,
        calculate_reserves_uncertainty,
        decline_curve_analysis,
    )
    from .facies import (
        FaciesClassifier,
        FaciesClassificationResult,
        FACIES_LABELS,
        load_facies_data,
        prepare_facies_features,
    )
    from .well_log_processor import (
        WellLogProcessor,
        ProcessedWellLogs,
        CurveQuality,
        CURVE_SIGNATURES,
        process_multiple_wells,
    )
    from .log_features import (
        LogFeatureEngineer,
        FeatureSet,
        prepare_ml_dataset,
    )
    from .formation_tops import (
        FormationTopDetector,
        FormationTop,
        BoundaryDetectionResult,
        compare_tops_with_reference,
    )
    from .confidence_scoring import (
        ConfidenceScorer,
        ConfidenceScore,
        WellConfidenceReport,
        compare_confidence_across_wells,
        export_review_list,
    )
    from .integration_exports import (
        LASExporter,
        FormationTopExporter,
        FaciesLogExporter,
        PetrelProjectExporter,
        create_correction_template,
        import_expert_corrections,
    )
    from .workflow_manager import (
        WorkflowManager,
        WorkflowIteration,
        CorrectionRecord,
        WorkflowState,
        create_workflow_dashboard,
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
    # Exceptions
    "exceptions",
]
