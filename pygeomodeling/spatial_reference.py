"""Coordinate Reference System (CRS) handling for spatial operations.

Provides standardized CRS handling using pyproj for all spatial operations
in the geomodeling toolkit.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

try:
    from pyproj import CRS, Transformer

    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    CRS = None  # type: ignore[assignment,misc]
    Transformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class SpatialReference:
    """Manages coordinate reference systems for spatial operations.

    Provides a unified interface for CRS handling across all spatial operations.
    """

    def __init__(self, crs: Union[str, int, CRS, None] = None):
        """Initialize spatial reference.

        Args:
            crs: Coordinate Reference System. Can be:
                - EPSG code (int or string like 'EPSG:32633')
                - CRS object from pyproj
                - Proj4 string
                - None (no CRS specified)

        Raises:
            ImportError: If pyproj is not available
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError(
                "pyproj is required for CRS handling. "
                "Install with: pip install pyproj or pip install pygeomodeling[geospatial]"
            )

        self._crs = None
        if crs is not None:
            self.crs = crs

    @property
    def crs(self) -> Optional[CRS]:
        """Get the CRS object."""
        return self._crs

    @crs.setter
    def crs(self, value: Union[str, int, CRS]) -> None:
        """Set the CRS.

        Args:
            value: CRS specification (EPSG code, CRS object, or Proj4 string)
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError("pyproj is required for CRS handling")

        self._crs = CRS.from_user_input(value)
        logger.info(f"CRS set to: {self._crs}")

    def transform(
        self,
        coordinates: np.ndarray,
        target_crs: Union[str, int, CRS],
        z_coordinate: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform coordinates to target CRS.

        Args:
            coordinates: Input coordinates [N, 2] or [N, 3] (x, y, [z])
            target_crs: Target CRS (EPSG code, CRS object, or string)
            z_coordinate: Optional Z coordinates if not included in coordinates array

        Returns:
            Transformed coordinates [N, 2] or [N, 3]

        Raises:
            ValueError: If source CRS is not set
        """
        if self._crs is None:
            raise ValueError("Source CRS not set. Set CRS before transforming coordinates.")

        target_crs_obj = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(self._crs, target_crs_obj, always_xy=True)

        # Handle 2D or 3D coordinates
        if coordinates.shape[1] == 2:
            x, y = coordinates[:, 0], coordinates[:, 1]
            z = z_coordinate if z_coordinate is not None else np.zeros(len(coordinates))
            x_new, y_new = transformer.transform(x, y)
            return np.column_stack([x_new, y_new, z])
        else:
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            x_new, y_new = transformer.transform(x, y)
            return np.column_stack([x_new, y_new, z])

    def get_units(self) -> str:
        """Get the units of the CRS.

        Returns:
            Unit string (e.g., 'metre', 'degree')
        """
        if self._crs is None:
            return "unknown"

        # Get axis units
        axis_info = self._crs.axis_info
        if axis_info:
            return axis_info[0].unit_name
        return "unknown"

    def get_epsg(self) -> Optional[int]:
        """Get EPSG code if available.

        Returns:
            EPSG code or None
        """
        if self._crs is None:
            return None

        try:
            return self._crs.to_epsg()
        except Exception:
            return None

    def __repr__(self) -> str:
        """String representation."""
        if self._crs is None:
            return "SpatialReference(crs=None)"
        epsg = self.get_epsg()
        if epsg:
            return f"SpatialReference(crs=EPSG:{epsg})"
        return f"SpatialReference(crs={self._crs})"


def transform_coordinates(
    coordinates: np.ndarray,
    source_crs: Union[str, int, CRS],
    target_crs: Union[str, int, CRS],
    z_coordinate: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transform coordinates between CRS.

    Convenience function for one-off transformations.

    Args:
        coordinates: Input coordinates [N, 2] or [N, 3]
        source_crs: Source CRS (EPSG code, CRS object, or string)
        target_crs: Target CRS (EPSG code, CRS object, or string)
        z_coordinate: Optional Z coordinates

    Returns:
        Transformed coordinates [N, 2] or [N, 3]

    Examples:
        >>> coords = np.array([[100000, 200000], [101000, 201000]])
        >>> # Transform from UTM Zone 33N to WGS84
        >>> coords_wgs84 = transform_coordinates(coords, 'EPSG:32633', 'EPSG:4326')
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError(
            "pyproj is required for coordinate transformation. "
            "Install with: pip install pyproj"
        )

    source_crs_obj = CRS.from_user_input(source_crs)
    target_crs_obj = CRS.from_user_input(target_crs)
    transformer = Transformer.from_crs(source_crs_obj, target_crs_obj, always_xy=True)

    if coordinates.shape[1] == 2:
        x, y = coordinates[:, 0], coordinates[:, 1]
        z = z_coordinate if z_coordinate is not None else np.zeros(len(coordinates))
        x_new, y_new = transformer.transform(x, y)
        return np.column_stack([x_new, y_new, z])
    else:
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        x_new, y_new = transformer.transform(x, y)
        return np.column_stack([x_new, y_new, z])


def get_crs_from_epsg(epsg_code: int) -> CRS:
    """Get CRS object from EPSG code.

    Args:
        epsg_code: EPSG code (e.g., 32633 for UTM Zone 33N)

    Returns:
        CRS object

    Examples:
        >>> crs = get_crs_from_epsg(32633)  # UTM Zone 33N
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required. Install with: pip install pyproj")

    return CRS.from_epsg(epsg_code)

