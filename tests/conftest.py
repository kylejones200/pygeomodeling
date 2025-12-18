"""Test configuration and fixtures for SPE9 geomodeling tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_grdecl_data():
    """Create sample GRDECL data for testing."""
    return {
        "dimensions": (10, 8, 5),  # Small test grid
        "properties": {
            "PERMX": np.random.lognormal(mean=2.0, sigma=1.5, size=(10, 8, 5)),
            "PERMY": np.random.lognormal(mean=2.0, sigma=1.5, size=(10, 8, 5)),
            "PERMZ": np.random.lognormal(mean=1.0, sigma=1.0, size=(10, 8, 5)),
            "PORO": np.random.uniform(0.1, 0.3, size=(10, 8, 5)),
        },
    }


@pytest.fixture
def sample_grdecl_file(tmp_path, sample_grdecl_data):
    """Create a temporary GRDECL file for testing."""
    grdecl_file = tmp_path / "test_spe9.grdecl"

    # Write a simple GRDECL format file
    with open(grdecl_file, "w") as f:
        nx, ny, nz = sample_grdecl_data["dimensions"]
        f.write(f"SPECGRID\n{nx} {ny} {nz} 1 F /\n\n")

        # Write PERMX data
        f.write("PERMX\n")
        permx_flat = sample_grdecl_data["properties"]["PERMX"].ravel()
        for i, val in enumerate(permx_flat):
            f.write(f"{val:.6f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n/\n\n")

        # Write PORO data
        f.write("PORO\n")
        poro_flat = sample_grdecl_data["properties"]["PORO"].ravel()
        for i, val in enumerate(poro_flat):
            f.write(f"{val:.6f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n/\n")

    return str(grdecl_file)


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.lognormal(mean=2.0, sigma=1.0, size=n_samples)
    return X, y


@pytest.fixture
def mock_spe9_toolkit():
    """Create a mock SPE9Toolkit for testing without real data."""
    from spe9_geomodeling.spe9_toolkit import GridData, SPE9Toolkit

    toolkit = SPE9Toolkit()

    # Mock data
    toolkit.data = {
        "dimensions": (5, 4, 3),
        "properties": {
            "PERMX": np.random.lognormal(mean=2.0, sigma=1.0, size=(5, 4, 3))
        },
    }

    # Mock grid data
    n_cells = 5 * 4 * 3
    X_grid = np.random.randn(n_cells, 3)  # x, y, z coordinates
    y_grid = np.random.lognormal(mean=2.0, sigma=1.0, size=n_cells)

    toolkit.grid_data = GridData(
        X_grid=X_grid,
        y_grid=y_grid,
        feature_names=["x_norm", "y_norm", "z_norm"],
        permx_3d=toolkit.data["properties"]["PERMX"],
        dimensions=(5, 4, 3),
    )

    return toolkit


@pytest.fixture
def skip_if_no_gpytorch():
    """Skip test if GPyTorch is not available."""
    try:
        import gpytorch
        import torch

        return True
    except ImportError:
        pytest.skip("GPyTorch not available")


@pytest.fixture
def skip_if_no_sklearn():
    """Skip test if scikit-learn is not available."""
    try:
        import sklearn

        return True
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


# Test data constants
TEST_DIMENSIONS = (10, 8, 5)
TEST_N_SAMPLES = 100
TEST_N_FEATURES = 5
TEST_RANDOM_STATE = 42

# Set random seeds for reproducible tests
np.random.seed(TEST_RANDOM_STATE)
