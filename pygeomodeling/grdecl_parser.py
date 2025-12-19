"""
GRDECL File Parser for Reservoir Modeling Data
Parses Eclipse-format GRDECL files to extract grid properties
"""

import logging
import re
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from .exceptions import (
    DataValidationError,
    raise_dimension_mismatch,
    raise_file_not_found,
    raise_invalid_format,
    raise_property_not_found,
)

# Configure logging
logger = logging.getLogger(__name__)


class GRDECLParser:
    def __init__(self, filepath: str):
        """Initialize GRDECL parser.

        Args:
            filepath: Path to GRDECL file

        Raises:
            DataValidationError: If filepath is invalid
        """
        if not filepath:
            raise DataValidationError(
                "Filepath cannot be empty",
                suggestion="Provide a valid path to a GRDECL file",
            )

        self.filepath = Path(filepath)
        self.grid_dimensions = None
        self.properties = {}

    def parse_specgrid(self, content: str) -> tuple[int, int, int]:
        """Parse SPECGRID keyword to get grid dimensions

        Args:
            content: GRDECL file content

        Returns:
            Tuple of (nx, ny, nz) grid dimensions

        Raises:
            FileFormatError: If SPECGRID keyword not found
        """
        specgrid_match = re.search(
            r"SPECGRID\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content, re.IGNORECASE
        )
        if specgrid_match:
            nx, ny, nz = map(int, specgrid_match.groups())
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise_invalid_format(
                    str(self.filepath),
                    "GRDECL",
                    f"Invalid grid dimensions: {nx} x {ny} x {nz}. All dimensions must be positive.",
                )
            return nx, ny, nz
        else:
            raise_invalid_format(
                str(self.filepath),
                "GRDECL",
                "SPECGRID keyword not found in file. This is required to define grid dimensions.",
            )

    def parse_property(self, content: str, property_name: str) -> np.ndarray:
        """Parse a property section (e.g., PERMX, PORO) from GRDECL content"""
        # Find the property keyword
        pattern = rf"{property_name}\s*\n(.*?)(?=\n[A-Z]|\n--|\Z)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if not match:
            raise ValueError(f"Property {property_name} not found in file")

        property_data = match.group(1)

        # Extract numerical values, handling comments and forward slashes
        numbers = []
        for line in property_data.split("\n"):
            # Remove comments (lines starting with --)
            line = re.sub(r"--.*", "", line)
            # Remove trailing / and whitespace
            line = re.sub(r"/.*", "", line)
            # Extract numbers
            line_numbers = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", line
            )
            numbers.extend([float(x) for x in line_numbers])

        return np.array(numbers)

    def load_data(self) -> dict:
        """Load and parse the GRDECL file

        Returns:
            Dictionary with 'dimensions' and 'properties' keys

        Raises:
            DataLoadError: If file cannot be loaded
            FileFormatError: If file format is invalid
        """
        if not self.filepath.exists():
            raise_file_not_found(str(self.filepath), "GRDECL")

        try:
            with open(self.filepath) as f:
                content = f.read()
        except PermissionError:
            raise DataValidationError(
                f"Permission denied reading file: {self.filepath}",
                suggestion="Check that you have read permissions for this file",
            )
        except Exception as e:
            raise DataValidationError(
                f"Error reading file: {self.filepath}\n{str(e)}",
                suggestion="Ensure the file is not corrupted and is a valid text file",
            )

        # Parse grid dimensions
        self.grid_dimensions = self.parse_specgrid(content)
        nx, ny, nz = self.grid_dimensions
        total_cells = nx * ny * nz

        logger.info("Grid dimensions: %d x %d x %d = %d cells", nx, ny, nz, total_cells)

        # Parse available properties
        properties_to_parse = ["PERMX", "PERMY", "PERMZ", "PORO", "NTG"]

        for prop in properties_to_parse:
            try:
                prop_data = self.parse_property(content, prop)
                if len(prop_data) == total_cells:
                    # Reshape to 3D array (Fortran order for reservoir modeling)
                    self.properties[prop] = prop_data.reshape((nx, ny, nz), order="F")
                    logger.info("Loaded %s: %d values", prop, len(prop_data))
                else:
                    raise_dimension_mismatch(
                        expected=(total_cells,),
                        actual=(len(prop_data),),
                        context=f"property {prop}",
                    )
            except ValueError as e:
                logger.warning("Could not load %s: %s", prop, e)

        if not self.properties:
            raise DataValidationError(
                "No properties were successfully loaded from the GRDECL file",
                suggestion=(
                    "Check that the file contains at least one of: "
                    f"{', '.join(properties_to_parse)}"
                ),
            )

        return {"dimensions": self.grid_dimensions, "properties": self.properties}

    def parse(self) -> dict:
        """Alias for load_data() for backward compatibility"""
        return self.load_data()

    def get_property_3d(self, property_name: str) -> Optional[np.ndarray]:
        """Get a 3D property array"""
        return self.properties.get(property_name)

    def get_property_slice(
        self, property_name: str, axis: str = "z", index: int = 0
    ) -> Optional[np.ndarray]:
        """Get a 2D slice of a property

        Args:
            property_name: Name of property to slice
            axis: Axis to slice along ('x', 'y', or 'z')
            index: Index along the axis

        Returns:
            2D slice of the property, or None if property not found

        Raises:
            PropertyNotFoundError: If property doesn't exist
            DataValidationError: If axis is invalid or index out of bounds
        """
        prop_3d = self.get_property_3d(property_name)
        if prop_3d is None:
            raise_property_not_found(property_name, list(self.properties.keys()))

        if axis.lower() not in ["x", "y", "z"]:
            raise DataValidationError(
                f"Invalid axis: {axis}", suggestion="axis must be 'x', 'y', or 'z'"
            )

        axis_sizes = {
            "x": prop_3d.shape[0],
            "y": prop_3d.shape[1],
            "z": prop_3d.shape[2],
        }
        max_index = axis_sizes[axis.lower()]

        if index < 0 or index >= max_index:
            raise DataValidationError(
                f"Index {index} out of bounds for axis '{axis}' (size: {max_index})",
                suggestion=f"Use an index between 0 and {max_index - 1}",
            )

        if axis.lower() == "z":
            return prop_3d[:, :, index]
        elif axis.lower() == "y":
            return prop_3d[:, index, :]
        else:  # x
            return prop_3d[index, :, :]


def load_spe9_data(
    data_path: Optional[str] = None,
):
    """Convenience function to load SPE9 dataset

    Args:
        data_path: Path to SPE9 GRDECL file. If None, uses the bundled data file.

    Returns:
        Dictionary containing parsed SPE9 data
    """
    if data_path is None:
        try:
            resource = resources.files("pygeomodeling.data") / "SPE9.GRDECL"
        except (FileNotFoundError, ModuleNotFoundError):
            raise_file_not_found(
                "SPE9.GRDECL",
                "bundled SPE9 dataset",
            )

        with resources.as_file(resource) as bundled_path:
            parser = GRDECLParser(str(bundled_path))
            return parser.load_data()

    parser = GRDECLParser(str(data_path))
    return parser.load_data()


if __name__ == "__main__":
    # Test the parser
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_spe9_data()
    logger.info("Available properties: %s", list(data["properties"].keys()))

    # Show some statistics
    for prop_name, prop_data in data["properties"].items():
        logger.info(
            "%s: min=%.2f, max=%.2f, mean=%.2f",
            prop_name,
            prop_data.min(),
            prop_data.max(),
            prop_data.mean(),
        )
