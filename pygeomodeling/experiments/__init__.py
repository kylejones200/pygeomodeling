"""
Experimental modules for advanced geomodeling research.

This subpackage contains experimental scripts and comparison frameworks
for evaluating different modeling approaches.
"""

try:
    from .deep_gp_experiment import DeepGPExperiment
except ImportError:
    # Handle case where GPyTorch dependencies aren't available
    pass

__all__ = ["DeepGPExperiment"]
