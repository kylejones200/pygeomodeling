"""
GRDECL File Parser for Reservoir Modeling Data
Parses Eclipse-format GRDECL files to extract grid properties
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


class GRDECLParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.grid_dimensions = None
        self.properties = {}

    def parse_specgrid(self, content: str) -> Tuple[int, int, int]:
        """Parse SPECGRID keyword to get grid dimensions"""
        specgrid_match = re.search(
            r"SPECGRID\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content, re.IGNORECASE
        )
        if specgrid_match:
            nx, ny, nz = map(int, specgrid_match.groups())
            return nx, ny, nz
        else:
            raise ValueError("SPECGRID not found in file")

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

    def load_data(self) -> Dict:
        """Load and parse the GRDECL file"""
        with open(self.filepath, "r") as f:
            content = f.read()

        # Parse grid dimensions
        self.grid_dimensions = self.parse_specgrid(content)
        nx, ny, nz = self.grid_dimensions
        total_cells = nx * ny * nz

        print(f"Grid dimensions: {nx} x {ny} x {nz} = {total_cells} cells")

        # Parse available properties
        properties_to_parse = ["PERMX", "PERMY", "PERMZ", "PORO", "NTG"]

        for prop in properties_to_parse:
            try:
                prop_data = self.parse_property(content, prop)
                if len(prop_data) == total_cells:
                    # Reshape to 3D array (Fortran order for reservoir modeling)
                    self.properties[prop] = prop_data.reshape((nx, ny, nz), order="F")
                    print(f"Loaded {prop}: {len(prop_data)} values")
                else:
                    print(
                        f"Warning: {prop} has {len(prop_data)} values, expected {total_cells}"
                    )
            except ValueError as e:
                print(f"Could not load {prop}: {e}")

        return {"dimensions": self.grid_dimensions, "properties": self.properties}

    def parse(self) -> Dict:
        """Alias for load_data() for backward compatibility"""
        return self.load_data()

    def get_property_3d(self, property_name: str) -> Optional[np.ndarray]:
        """Get a 3D property array"""
        return self.properties.get(property_name)

    def get_property_slice(
        self, property_name: str, axis: str = "z", index: int = 0
    ) -> Optional[np.ndarray]:
        """Get a 2D slice of a property"""
        prop_3d = self.get_property_3d(property_name)
        if prop_3d is None:
            return None

        if axis.lower() == "z":
            return prop_3d[:, :, index]
        elif axis.lower() == "y":
            return prop_3d[:, index, :]
        elif axis.lower() == "x":
            return prop_3d[index, :, :]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")


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
        # Use the bundled data file in the project
        module_dir = Path(__file__).parent.parent
        data_path = module_dir / "data" / "SPE9.GRDECL"

    parser = GRDECLParser(str(data_path))
    return parser.load_data()


if __name__ == "__main__":
    # Test the parser
    data = load_spe9_data()
    print("\nAvailable properties:", list(data["properties"].keys()))

    # Show some statistics
    for prop_name, prop_data in data["properties"].items():
        print(
            f"{prop_name}: min={prop_data.min():.2f}, max={prop_data.max():.2f}, mean={prop_data.mean():.2f}"
        )
