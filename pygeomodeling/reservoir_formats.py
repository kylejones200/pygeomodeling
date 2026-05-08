"""Reservoir format readers for multiple industry standards.

Supports:
- GRDECL (Eclipse format) - already implemented
- Petrel ASCII grid format
- Petrel binary format
- RESQML (via h5py)
- ASCII grid formats (Surfer, GSLIB, etc.)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from pyproj import CRS, Transformer

    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    CRS = None  # type: ignore[assignment,misc]
    Transformer = None  # type: ignore[assignment,misc]

from .exceptions import DataValidationError, raise_file_not_found, raise_invalid_format
from .grdecl_parser import GRDECLParser

logger = logging.getLogger(__name__)


@dataclass
class GridMetadata:
    """Metadata for reservoir grid data.

    Attributes:
        dimensions: Grid dimensions (nx, ny, nz)
        origin: Origin coordinates (x0, y0, z0)
        cell_size: Cell size (dx, dy, dz)
        rotation: Rotation angle in degrees (default: 0)
        crs: Coordinate Reference System (EPSG code or CRS object)
        properties: Dictionary of property names and their metadata
    """

    dimensions: tuple[int, int, int]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cell_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: float = 0.0
    crs: Optional[Any] = None  # CRS object or EPSG code
    properties: dict[str, dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class ReservoirGrid:
    """Container for reservoir grid data.

    Attributes:
        properties: Dictionary of property arrays {name: array}
        metadata: Grid metadata
        coordinates: Optional coordinate arrays (x, y, z) for each cell
    """

    properties: dict[str, np.ndarray]
    metadata: GridMetadata
    coordinates: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate grid data."""
        nx, ny, nz = self.metadata.dimensions
        expected_size = nx * ny * nz

        for prop_name, prop_data in self.properties.items():
            if prop_data.size != expected_size:
                raise DataValidationError(
                    f"Property '{prop_name}' size {prop_data.size} "
                    f"does not match grid dimensions {expected_size}",
                    suggestion="Check grid dimensions and property data",
                )

    def get_property(self, name: str) -> np.ndarray:
        """Get a property array, reshaped to 3D if needed."""
        prop = self.properties[name]
        if prop.ndim == 1:
            nx, ny, nz = self.metadata.dimensions
            return prop.reshape(nx, ny, nz)
        return prop

    def transform_coordinates(
        self, target_crs: str | int | Any, in_place: bool = False
    ) -> Optional[np.ndarray]:
        """Transform coordinates to target CRS.

        Args:
            target_crs: Target CRS (EPSG code, CRS object, or string)
            in_place: If True, update coordinates in place

        Returns:
            Transformed coordinates or None if in_place=True
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError(
                "pyproj is required for coordinate transformation. "
                "Install with: pip install pyproj"
            )

        if self.coordinates is None:
            raise ValueError("No coordinates available to transform")

        if self.metadata.crs is None:
            raise ValueError("Source CRS not specified in metadata")

        # Create transformer
        source_crs = CRS.from_user_input(self.metadata.crs)
        target_crs_obj = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)

        # Transform coordinates
        coords_flat = self.coordinates.reshape(-1, 3)
        x, y, z = coords_flat[:, 0], coords_flat[:, 1], coords_flat[:, 2]
        x_new, y_new = transformer.transform(x, y)
        z_new = z  # Z typically doesn't transform

        transformed = np.column_stack([x_new, y_new, z_new])
        transformed = transformed.reshape(self.coordinates.shape)

        if in_place:
            self.coordinates = transformed
            self.metadata.crs = target_crs
            return None
        else:
            return transformed


class PetrelASCIIReader:
    """Reader for Petrel ASCII grid format.

    Petrel ASCII format typically has:
    - Header with dimensions and grid info
    - Property data in column format or grid format
    """

    def __init__(self, filepath: str | Path):
        """Initialize Petrel ASCII reader.

        Args:
            filepath: Path to Petrel ASCII file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise_file_not_found(str(self.filepath))

    def read(self, property_names: Optional[list[str]] = None) -> ReservoirGrid:
        """Read Petrel ASCII grid file.

        Args:
            property_names: Optional list of property names to read.
                           If None, reads all properties found.

        Returns:
            ReservoirGrid object with properties and metadata
        """
        with open(self.filepath, "r") as f:
            content = f.read()

        lines = content.split("\n")
        header_lines = []
        data_start = 0

        # Parse header (typically first 10-20 lines)
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue

            # Look for dimension keywords
            if any(keyword in line_stripped.upper() for keyword in ["NX", "NY", "NZ", "DIM"]):
                header_lines.append((i, line_stripped))
            elif line_stripped.replace(".", "").replace("-", "").isdigit():
                data_start = i
                break

        # Extract dimensions
        nx, ny, nz = self._parse_dimensions(header_lines, lines)

        # Parse data section
        properties = {}
        if property_names is None:
            # Try to detect property names from header or use defaults
            property_names = self._detect_property_names(header_lines, lines)

        # Read property data
        data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith("#")]
        data_array = np.array([list(map(float, line.split())) for line in data_lines if line])

        # Reshape based on format (column or grid)
        if data_array.shape[1] > 3:  # Column format: x, y, z, prop1, prop2, ...
            coords = data_array[:, :3]
            for i, prop_name in enumerate(property_names):
                if i + 3 < data_array.shape[1]:
                    prop_data = data_array[:, i + 3]
                    # Reshape to grid if possible
                    if prop_data.size == nx * ny * nz:
                        properties[prop_name] = prop_data.reshape(nx, ny, nz)
                    else:
                        properties[prop_name] = prop_data
        else:  # Grid format
            # Assume single property in grid format
            if data_array.size == nx * ny * nz:
                properties[property_names[0] if property_names else "property"] = data_array.flatten()

        # Create metadata
        metadata = GridMetadata(dimensions=(nx, ny, nz))

        return ReservoirGrid(properties=properties, metadata=metadata)

    def _parse_dimensions(self, header_lines: list, all_lines: list) -> tuple[int, int, int]:
        """Parse grid dimensions from header."""
        # Try common Petrel dimension formats
        for i, line in header_lines:
            line_upper = line.upper()
            if "NX" in line_upper and "NY" in line_upper and "NZ" in line_upper:
                # Format: NX=24 NY=25 NZ=15
                parts = line_upper.split()
                nx = ny = nz = None
                for part in parts:
                    if "NX" in part:
                        nx = int(part.split("=")[-1])
                    elif "NY" in part:
                        ny = int(part.split("=")[-1])
                    elif "NZ" in part:
                        nz = int(part.split("=")[-1])
                if nx and ny and nz:
                    return (nx, ny, nz)

        # Fallback: try to infer from data
        raise raise_invalid_format(
            str(self.filepath),
            "Petrel ASCII",
            "Could not parse grid dimensions from header",
        )

    def _detect_property_names(self, header_lines: list, all_lines: list) -> list[str]:
        """Detect property names from header."""
        # Look for property name keywords
        property_names = []
        for i, line in header_lines:
            if "PROPERTY" in line.upper() or "PROP" in line.upper():
                # Extract property names
                parts = line.split()
                for part in parts:
                    if part.upper() not in ["PROPERTY", "PROP", "NAME"]:
                        property_names.append(part)

        return property_names if property_names else ["property"]


class PetrelBinaryReader:
    """Reader for Petrel binary grid format.

    Petrel binary format is typically:
    - Header with metadata
    - Binary data in float32 or float64 format
    """

    def __init__(self, filepath: str | Path):
        """Initialize Petrel binary reader.

        Args:
            filepath: Path to Petrel binary file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise_file_not_found(str(self.filepath))

    def read(
        self,
        dimensions: tuple[int, int, int],
        dtype: type = np.float32,
        byte_order: str = "<",
    ) -> ReservoirGrid:
        """Read Petrel binary grid file.

        Args:
            dimensions: Grid dimensions (nx, ny, nz)
            dtype: Data type (default: float32)
            byte_order: Byte order ('<' for little-endian, '>' for big-endian)

        Returns:
            ReservoirGrid object with properties and metadata
        """
        nx, ny, nz = dimensions
        expected_size = nx * ny * nz

        with open(self.filepath, "rb") as f:
            data = np.frombuffer(f.read(), dtype=f"{byte_order}{dtype().dtype.char}")
            if data.size != expected_size:
                raise DataValidationError(
                    f"File size {data.size} does not match expected {expected_size}",
                    suggestion=f"Check dimensions: {dimensions}",
                )

        # Reshape to 3D
        property_data = data.reshape(nx, ny, nz)

        metadata = GridMetadata(dimensions=dimensions)
        return ReservoirGrid(
            properties={"property": property_data.flatten()}, metadata=metadata
        )


class RESQMLReader:
    """Reader for RESQML format (via HDF5).

    RESQML is an industry standard for reservoir data exchange.
    This reader handles HDF5-based RESQML files.
    """

    def __init__(self, filepath: str | Path):
        """Initialize RESQML reader.

        Args:
            filepath: Path to RESQML HDF5 file
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for RESQML support. "
                "Install with: pip install pygeomodeling[geospatial]"
            )

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise_file_not_found(str(self.filepath))

    def read(self, property_path: Optional[str] = None) -> ReservoirGrid:
        """Read RESQML HDF5 file.

        Args:
            property_path: Optional path to property in HDF5 file.
                          If None, attempts to auto-detect.

        Returns:
            ReservoirGrid object with properties and metadata
        """
        with h5py.File(self.filepath, "r") as f:
            # RESQML structure varies, try common paths
            if property_path:
                prop_data = np.array(f[property_path])
            else:
                # Auto-detect: look for common RESQML groups
                prop_data, dimensions = self._auto_detect_property(f)

            # Extract dimensions if available
            if "dimensions" in f.attrs:
                dims = f.attrs["dimensions"]
                if len(dims) == 3:
                    dimensions = tuple(dims)
                else:
                    dimensions = prop_data.shape[:3] if prop_data.ndim >= 3 else (1, 1, prop_data.size)
            else:
                dimensions = prop_data.shape[:3] if prop_data.ndim >= 3 else (1, 1, prop_data.size)

            # Extract metadata
            metadata = self._extract_metadata(f, dimensions)

            # Reshape if needed
            nx, ny, nz = dimensions
            if prop_data.size == nx * ny * nz:
                prop_data = prop_data.reshape(nx, ny, nz)

            return ReservoirGrid(
                properties={"property": prop_data.flatten()}, metadata=metadata
            )

    def _auto_detect_property(self, h5_file: h5py.File) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Auto-detect property data in RESQML file."""
        # Common RESQML paths
        common_paths = [
            "/RESQML/Grid/Property",
            "/Grid/Property",
            "/Property",
            "/values",
        ]

        for path in common_paths:
            if path in h5_file:
                data = np.array(h5_file[path])
                # Try to infer dimensions
                if data.ndim == 3:
                    dimensions = data.shape
                elif data.ndim == 1:
                    # Try to infer from other metadata
                    dimensions = (1, 1, data.size)
                else:
                    dimensions = data.shape[:3] if data.ndim >= 3 else (1, 1, data.size)
                return data, dimensions

        # Fallback: get first dataset
        datasets = []

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(obj)

        h5_file.visititems(collect_datasets)
        if datasets:
            dataset = datasets[0]
            data = np.array(dataset)
            dimensions = data.shape[:3] if data.ndim >= 3 else (1, 1, data.size)
            return data, dimensions

        raise raise_invalid_format(
            str(self.filepath), "RESQML", "Could not find property data in HDF5 file"
        )

    def _extract_metadata(self, h5_file: h5py.File, dimensions: tuple[int, int, int]) -> GridMetadata:
        """Extract metadata from RESQML file."""
        metadata = GridMetadata(dimensions=dimensions)

        # Try to extract CRS
        if "crs" in h5_file.attrs:
            metadata.crs = h5_file.attrs["crs"]
        elif "EPSG" in h5_file.attrs:
            metadata.crs = f"EPSG:{h5_file.attrs['EPSG']}"

        # Extract origin and cell size if available
        if "origin" in h5_file.attrs:
            metadata.origin = tuple(h5_file.attrs["origin"])
        if "cell_size" in h5_file.attrs:
            metadata.cell_size = tuple(h5_file.attrs["cell_size"])

        return metadata


class ASCIIGridReader:
    """Reader for various ASCII grid formats (Surfer, GSLIB, etc.).

    Supports:
    - Surfer ASCII grid (.grd)
    - GSLIB format
    - Simple column-based formats
    """

    def __init__(self, filepath: str | Path, format: str = "auto"):
        """Initialize ASCII grid reader.

        Args:
            filepath: Path to ASCII grid file
            format: Format type ('surfer', 'gslib', 'column', 'auto')
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise_file_not_found(str(self.filepath))
        self.format = format

    def read(self) -> ReservoirGrid:
        """Read ASCII grid file.

        Returns:
            ReservoirGrid object with properties and metadata
        """
        if self.format == "auto":
            self.format = self._detect_format()

        if self.format == "surfer":
            return self._read_surfer()
        elif self.format == "gslib":
            return self._read_gslib()
        elif self.format == "column":
            return self._read_column()
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _detect_format(self) -> str:
        """Auto-detect file format."""
        with open(self.filepath, "r") as f:
            first_lines = [f.readline().strip() for _ in range(5)]

        # Check for Surfer format (starts with DSAA)
        if any("DSAA" in line for line in first_lines):
            return "surfer"

        # Check for GSLIB format (starts with dimensions)
        if first_lines[0].isdigit():
            return "gslib"

        # Default to column format
        return "column"

    def _read_surfer(self) -> ReservoirGrid:
        """Read Surfer ASCII grid format."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # Surfer format: DSAA header
        header_idx = None
        for i, line in enumerate(lines):
            if "DSAA" in line.upper():
                header_idx = i
                break

        if header_idx is None:
            raise raise_invalid_format(str(self.filepath), "Surfer", "DSAA header not found")

        # Parse header
        nx = int(lines[header_idx + 1].split()[0])
        ny = int(lines[header_idx + 1].split()[1])
        xmin, xmax = map(float, lines[header_idx + 2].split())
        ymin, ymax = map(float, lines[header_idx + 3].split())
        zmin, zmax = map(float, lines[header_idx + 4].split())

        # Read data
        data_lines = [l.strip() for l in lines[header_idx + 5:] if l.strip()]
        data = np.array([float(x) for line in data_lines for x in line.split()])

        # Reshape (Surfer uses row-major, need to flip)
        data = data.reshape(ny, nx)
        data = np.flipud(data)  # Flip vertically
        data = data.T  # Transpose

        metadata = GridMetadata(
            dimensions=(nx, ny, 1),
            origin=(xmin, ymin, 0.0),
            cell_size=((xmax - xmin) / nx, (ymax - ymin) / ny, 1.0),
        )

        return ReservoirGrid(properties={"property": data.flatten()}, metadata=metadata)

    def _read_gslib(self) -> ReservoirGrid:
        """Read GSLIB format."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # GSLIB format: first line is number of variables
        n_vars = int(lines[0].strip())
        var_names = [lines[i].strip() for i in range(1, n_vars + 1)]

        # Read data
        data_lines = [l.strip() for l in lines[n_vars + 1:] if l.strip()]
        data = np.array([list(map(float, line.split())) for line in data_lines])

        # Assume 2D grid (x, y, properties...)
        n_points = len(data)
        # Try to infer grid dimensions
        nx = int(np.sqrt(n_points))
        ny = nx
        if nx * ny != n_points:
            # Not a perfect square, use as 1D
            nx, ny = n_points, 1

        properties = {}
        for i, var_name in enumerate(var_names):
            if i < data.shape[1]:
                prop_data = data[:, i]
                if prop_data.size == nx * ny:
                    properties[var_name] = prop_data.reshape(nx, ny, 1).flatten()
                else:
                    properties[var_name] = prop_data

        metadata = GridMetadata(dimensions=(nx, ny, 1))
        return ReservoirGrid(properties=properties, metadata=metadata)

    def _read_column(self) -> ReservoirGrid:
        """Read simple column-based format."""
        data = np.loadtxt(self.filepath)

        if data.ndim == 1:
            # Single column
            properties = {"property": data}
            dimensions = (data.size, 1, 1)
        else:
            # Multiple columns: assume first 3 are x, y, z, rest are properties
            if data.shape[1] >= 4:
                properties = {}
                for i in range(3, data.shape[1]):
                    properties[f"property_{i-2}"] = data[:, i]
                dimensions = (data.shape[0], 1, 1)
            else:
                properties = {"property": data.flatten()}
                dimensions = data.shape if data.ndim >= 2 else (data.size, 1, 1)

        metadata = GridMetadata(dimensions=dimensions)
        return ReservoirGrid(properties=properties, metadata=metadata)


def load_reservoir_data(
    filepath: str | Path,
    format: Optional[str] = None,
    crs: Optional[str | int] = None,
    **kwargs,
) -> ReservoirGrid:
    """Unified function to load reservoir data from various formats.

    Args:
        filepath: Path to reservoir data file
        format: Format type ('grdecl', 'petrel_ascii', 'petrel_binary', 'resqml', 'ascii_grid', 'auto')
        crs: Optional coordinate reference system (EPSG code or CRS string)
        **kwargs: Additional format-specific arguments

    Returns:
        ReservoirGrid object

    Examples:
        >>> # Auto-detect format
        >>> grid = load_reservoir_data('data/reservoir.grdecl')
        >>> # Specify format and CRS
        >>> grid = load_reservoir_data('data/petrel.asc', format='petrel_ascii', crs='EPSG:32633')
        >>> # RESQML with property path
        >>> grid = load_reservoir_data('data/resqml.h5', format='resqml', property_path='/Grid/Property')
    """
    filepath = Path(filepath)

    # Auto-detect format if not specified
    if format is None or format == "auto":
        format = _detect_file_format(filepath)

    # Load based on format
    if format == "grdecl":
        parser = GRDECLParser(str(filepath))
        data = parser.parse()
        properties = data["properties"]
        dimensions = data["dimensions"]

        metadata = GridMetadata(dimensions=dimensions, crs=crs)
        return ReservoirGrid(properties=properties, metadata=metadata)

    elif format == "petrel_ascii":
        reader = PetrelASCIIReader(filepath)
        grid = reader.read(**kwargs)
        if crs:
            grid.metadata.crs = crs
        return grid

    elif format == "petrel_binary":
        dimensions = kwargs.pop("dimensions", (24, 25, 15))
        reader = PetrelBinaryReader(filepath)
        grid = reader.read(dimensions=dimensions, **kwargs)
        if crs:
            grid.metadata.crs = crs
        return grid

    elif format == "resqml":
        reader = RESQMLReader(filepath)
        grid = reader.read(**kwargs)
        if crs:
            grid.metadata.crs = crs
        return grid

    elif format == "ascii_grid":
        reader = ASCIIGridReader(filepath, format=kwargs.pop("ascii_format", "auto"))
        grid = reader.read()
        if crs:
            grid.metadata.crs = crs
        return grid

    else:
        raise ValueError(
            f"Unknown format: {format}. "
            "Choose from: 'grdecl', 'petrel_ascii', 'petrel_binary', 'resqml', 'ascii_grid'"
        )


def _detect_file_format(filepath: Path) -> str:
    """Auto-detect file format from extension and content."""
    ext = filepath.suffix.lower()

    # Check extension
    if ext == ".grdecl":
        return "grdecl"
    elif ext in [".asc", ".ascii", ".txt"]:
        # Check content for Petrel vs generic ASCII
        with open(filepath, "r") as f:
            first_line = f.readline()
            if "NX" in first_line.upper() or "PETREL" in first_line.upper():
                return "petrel_ascii"
            else:
                return "ascii_grid"
    elif ext == ".bin" or ext == ".petrel":
        return "petrel_binary"
    elif ext in [".h5", ".hdf5", ".resqml"]:
        return "resqml"
    else:
        # Try to read as ASCII grid
        return "ascii_grid"

