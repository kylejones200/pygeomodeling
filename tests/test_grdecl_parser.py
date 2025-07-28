"""Tests for GRDECL parser functionality."""

import pytest
import numpy as np
from pathlib import Path

from spe9_geomodeling.grdecl_parser import GRDECLParser, load_spe9_data


class TestGRDECLParser:
    """Test cases for GRDECLParser class."""

    def test_parser_initialization(self, sample_grdecl_file):
        """Test parser initialization."""
        parser = GRDECLParser(sample_grdecl_file)
        assert parser.filepath == sample_grdecl_file
        assert parser.grid_dimensions is None
        assert parser.properties == {}

    def test_parse_file(self, sample_grdecl_file, sample_grdecl_data):
        """Test parsing a GRDECL file."""
        parser = GRDECLParser(sample_grdecl_file)
        data = parser.parse()

        # Check dimensions
        assert data["dimensions"] == sample_grdecl_data["dimensions"]

        # Check that PERMX was parsed
        assert "PERMX" in data["properties"]
        assert data["properties"]["PERMX"].shape == sample_grdecl_data["dimensions"]

        # Check that PORO was parsed
        assert "PORO" in data["properties"]
        assert data["properties"]["PORO"].shape == sample_grdecl_data["dimensions"]

    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file raises appropriate error."""
        parser = GRDECLParser("nonexistent_file.grdecl")
        with pytest.raises(FileNotFoundError):
            parser.parse()

    def test_parse_invalid_file(self, tmp_path):
        """Test parsing an invalid GRDECL file."""
        invalid_file = tmp_path / "invalid.grdecl"
        invalid_file.write_text("This is not a valid GRDECL file")

        parser = GRDECLParser(str(invalid_file))
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, KeyError)):
            parser.parse()


class TestLoadSPE9Data:
    """Test cases for load_spe9_data function."""

    def test_load_spe9_data_with_file(self, sample_grdecl_file):
        """Test loading SPE9 data from file."""
        data = load_spe9_data(sample_grdecl_file)

        assert "dimensions" in data
        assert "properties" in data
        assert isinstance(data["dimensions"], tuple)
        assert len(data["dimensions"]) == 3
        assert isinstance(data["properties"], dict)

    def test_load_spe9_data_default_path(self):
        """Test loading SPE9 data with default path."""
        # This test will skip if the default file doesn't exist
        try:
            data = load_spe9_data()
            assert "dimensions" in data
            assert "properties" in data
        except FileNotFoundError:
            pytest.skip("Default SPE9 data file not found")

    def test_load_spe9_data_invalid_path(self):
        """Test loading SPE9 data with invalid path."""
        with pytest.raises(FileNotFoundError):
            load_spe9_data("nonexistent_path.grdecl")


class TestGRDECLDataValidation:
    """Test data validation and properties."""

    def test_permx_properties(self, sample_grdecl_file):
        """Test PERMX data properties."""
        data = load_spe9_data(sample_grdecl_file)
        permx = data["properties"]["PERMX"]

        # Check basic properties
        assert permx.ndim == 3
        assert permx.shape == data["dimensions"]
        assert np.all(permx >= 0)  # Permeability should be non-negative
        assert not np.all(permx == 0)  # Should have some variation

    def test_porosity_properties(self, sample_grdecl_file):
        """Test porosity data properties."""
        data = load_spe9_data(sample_grdecl_file)

        if "PORO" in data["properties"]:
            poro = data["properties"]["PORO"]

            # Check basic properties
            assert poro.ndim == 3
            assert poro.shape == data["dimensions"]
            assert np.all(poro >= 0)  # Porosity should be non-negative
            assert np.all(poro <= 1)  # Porosity should be <= 1

    def test_data_consistency(self, sample_grdecl_file):
        """Test consistency between different properties."""
        data = load_spe9_data(sample_grdecl_file)
        dimensions = data["dimensions"]

        # All properties should have the same shape
        for prop_name, prop_data in data["properties"].items():
            assert prop_data.shape == dimensions, f"{prop_name} has incorrect shape"
            assert not np.any(np.isnan(prop_data)), f"{prop_name} contains NaN values"
            assert not np.any(
                np.isinf(prop_data)
            ), f"{prop_name} contains infinite values"


class TestGRDECLParserEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        empty_file = tmp_path / "empty.grdecl"
        empty_file.write_text("")

        parser = GRDECLParser(str(empty_file))
        with pytest.raises((ValueError, KeyError)):
            parser.parse()

    def test_malformed_dimensions(self, tmp_path):
        """Test parsing file with malformed dimensions."""
        malformed_file = tmp_path / "malformed.grdecl"
        malformed_file.write_text("SPECGRID\nINVALID DIMENSIONS\n/\n")

        parser = GRDECLParser(str(malformed_file))
        with pytest.raises((ValueError, IndexError)):
            parser.parse()

    def test_missing_properties(self, tmp_path):
        """Test parsing file with missing required properties."""
        minimal_file = tmp_path / "minimal.grdecl"
        minimal_file.write_text("SPECGRID\n5 5 5 1 F /\n")

        parser = GRDECLParser(str(minimal_file))
        data = parser.parse()

        # Should parse dimensions but have empty properties
        assert data["dimensions"] == (5, 5, 5)
        # Properties might be empty or have default values
        assert isinstance(data["properties"], dict)
