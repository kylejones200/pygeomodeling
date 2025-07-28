"""Basic tests for the spe9_geomodeling package."""

import pytest
import numpy as np
import sys
from pathlib import Path


class TestPackageBasics:
    """Basic package-level tests."""
    
    def test_numpy_import(self):
        """Test that numpy can be imported and basic operations work."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert len(arr) == 5
    
    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import spe9_geomodeling
            assert hasattr(spe9_geomodeling, '__version__')
            assert hasattr(spe9_geomodeling, '__author__')
        except ImportError:
            pytest.skip("Package not installed in development mode")
    
    def test_package_structure(self):
        """Test that package has expected structure."""
        try:
            import spe9_geomodeling
            
            # Test that main classes are available
            expected_classes = [
                'SPE9Toolkit', 'UnifiedSPE9Toolkit', 'SPE9Plotter',
                'GRDECLParser', 'load_spe9_data'
            ]
            
            for class_name in expected_classes:
                assert hasattr(spe9_geomodeling, class_name), f"Missing {class_name}"
                
        except ImportError:
            pytest.skip("Package not installed in development mode")
    
    def test_optional_dependencies(self):
        """Test handling of optional dependencies."""
        try:
            import spe9_geomodeling
            
            # GPyTorch classes should be available if GPyTorch is installed
            try:
                import torch
                import gpytorch
                # If GPyTorch is available, these should be importable
                assert hasattr(spe9_geomodeling, 'SPE9GPModel')
                assert hasattr(spe9_geomodeling, 'DeepGPModel')
            except ImportError:
                # GPyTorch not available, classes might not be in __all__
                pass
                
        except ImportError:
            pytest.skip("Package not installed in development mode")
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        try:
            import spe9_geomodeling
            version = spe9_geomodeling.__version__
            
            # Basic check for semantic versioning format (X.Y.Z)
            parts = version.split('.')
            assert len(parts) >= 2, f"Version {version} doesn't follow semantic versioning"
            
            # Check that parts are numeric
            for part in parts[:3]:  # Major.Minor.Patch
                assert part.isdigit() or part.replace('-', '').replace('+', '').split('.')[0].isdigit()
                
        except ImportError:
            pytest.skip("Package not installed in development mode")


class TestDependencies:
    """Test package dependencies."""
    
    def test_required_dependencies(self):
        """Test that required dependencies are available."""
        required_packages = [
            'numpy', 'scipy', 'scikit-learn', 'matplotlib', 
            'seaborn', 'pandas', 'pathlib'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required dependency {package} not available")
    
    def test_optional_dependencies_handling(self):
        """Test that optional dependencies are handled gracefully."""
        # Test GPyTorch availability
        try:
            import torch
            import gpytorch
            gpytorch_available = True
        except ImportError:
            gpytorch_available = False
        
        # Package should import regardless of GPyTorch availability
        try:
            import spe9_geomodeling
            assert True
        except ImportError:
            pytest.fail("Package import failed due to optional dependency issues")


class TestBasicFunctionality:
    """Test basic functionality works."""
    
    def test_basic_math(self):
        """Test basic mathematical operations."""
        assert 1 + 1 == 2
        assert np.pi > 3.0
        assert np.e > 2.0
    
    def test_numpy_operations(self):
        """Test numpy operations work correctly."""
        # Test array creation and operations
        arr = np.random.randn(100)
        assert len(arr) == 100
        assert isinstance(arr.mean(), (float, np.floating))
        assert isinstance(arr.std(), (float, np.floating))
        
        # Test 3D array operations (relevant for geomodeling)
        arr_3d = np.random.randn(10, 8, 5)
        assert arr_3d.shape == (10, 8, 5)
        assert arr_3d.ndim == 3
    
    def test_path_operations(self):
        """Test path operations work correctly."""
        from pathlib import Path
        
        # Test basic path operations
        test_path = Path("/tmp/test_file.txt")
        assert test_path.name == "test_file.txt"
        assert test_path.suffix == ".txt"
        assert test_path.stem == "test_file"


class TestEnvironment:
    """Test environment and system requirements."""
    
    def test_python_version(self):
        """Test Python version is supported."""
        # Package requires Python 3.9+
        assert sys.version_info >= (3, 9), f"Python {sys.version_info} is too old"
    
    def test_working_directory(self):
        """Test that tests can access working directory."""
        cwd = Path.cwd()
        assert cwd.exists()
        assert cwd.is_dir()
    
    def test_temp_directory_access(self):
        """Test that temporary directory is accessible."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists()
            assert temp_path.is_dir()
            
            # Test file creation in temp directory
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"


if __name__ == "__main__":
    pytest.main([__file__])
