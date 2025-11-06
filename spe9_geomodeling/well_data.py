"""
Well Data Integration Module

Handles LAS file parsing, well log processing, and integration with reservoir grids.
Supports standard LAS 2.0 format used in the oil & gas industry.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from .exceptions import DataLoadError, FileFormatError, DataValidationError


@dataclass
class WellHeader:
    """Well header information from LAS file."""
    well_name: str
    uwi: Optional[str] = None  # Unique Well Identifier
    field: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    operator: Optional[str] = None
    api: Optional[str] = None
    null_value: float = -999.25
    
    def __str__(self) -> str:
        return (
            f"Well: {self.well_name}\n"
            f"  UWI: {self.uwi}\n"
            f"  Field: {self.field}\n"
            f"  Operator: {self.operator}"
        )


@dataclass
class CurveInfo:
    """Information about a log curve."""
    mnemonic: str
    unit: str
    description: str
    
    def __str__(self) -> str:
        return f"{self.mnemonic} ({self.unit}): {self.description}"


class LASParser:
    """
    Parser for LAS (Log ASCII Standard) files.
    
    Supports LAS 2.0 format commonly used for well log data.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize LAS parser.
        
        Args:
            filepath: Path to LAS file
        """
        self.filepath = Path(filepath)
        self.header = None
        self.curves = {}
        self.data = None
        
        if not self.filepath.exists():
            raise DataLoadError(
                f"LAS file not found: {filepath}",
                suggestion="Check file path and permissions"
            )
    
    def parse(self) -> pd.DataFrame:
        """
        Parse LAS file and return data as DataFrame.
        
        Returns:
            DataFrame with depth index and log curves as columns
        """
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse sections
        self._parse_version_section(content)
        self._parse_well_section(content)
        self._parse_curve_section(content)
        self._parse_data_section(content)
        
        return self.data
    
    def _parse_version_section(self, content: str):
        """Parse ~V (Version) section."""
        version_match = re.search(r'~V.*?(?=~[A-Z])', content, re.DOTALL)
        if version_match:
            version_text = version_match.group(0)
            # Extract version number
            vers_match = re.search(r'VERS\s*\.\s*(\d+\.?\d*)', version_text)
            if vers_match:
                version = float(vers_match.group(1))
                if version > 2.0:
                    raise FileFormatError(
                        f"LAS version {version} not fully supported",
                        suggestion="This parser supports LAS 2.0"
                    )
    
    def _parse_well_section(self, content: str):
        """Parse ~W (Well Information) section."""
        well_match = re.search(r'~W.*?(?=~[A-Z])', content, re.DOTALL)
        if not well_match:
            raise FileFormatError(
                "No ~W (Well) section found in LAS file",
                suggestion="Ensure file is valid LAS format"
            )
        
        well_text = well_match.group(0)
        
        # Extract well parameters
        well_name = self._extract_parameter(well_text, 'WELL')
        uwi = self._extract_parameter(well_text, 'UWI')
        field = self._extract_parameter(well_text, 'FLD')
        location = self._extract_parameter(well_text, 'LOC')
        country = self._extract_parameter(well_text, 'CTRY')
        operator = self._extract_parameter(well_text, 'COMP')
        api = self._extract_parameter(well_text, 'API')
        
        null_str = self._extract_parameter(well_text, 'NULL')
        null_value = float(null_str) if null_str else -999.25
        
        self.header = WellHeader(
            well_name=well_name or "Unknown",
            uwi=uwi,
            field=field,
            location=location,
            country=country,
            operator=operator,
            api=api,
            null_value=null_value
        )
    
    def _parse_curve_section(self, content: str):
        """Parse ~C (Curve Information) section."""
        curve_match = re.search(r'~C.*?(?=~[A-Z])', content, re.DOTALL)
        if not curve_match:
            raise FileFormatError(
                "No ~C (Curve) section found",
                suggestion="LAS file must define curves"
            )
        
        curve_text = curve_match.group(0)
        
        # Parse each curve definition
        # Format: MNEM.UNIT  VALUE : DESCRIPTION
        curve_pattern = r'(\w+)\s*\.\s*(\S+)\s+[^:]*:\s*(.+)'
        
        for line in curve_text.split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            match = re.match(curve_pattern, line)
            if match:
                mnemonic = match.group(1)
                unit = match.group(2)
                description = match.group(3).strip()
                
                self.curves[mnemonic] = CurveInfo(
                    mnemonic=mnemonic,
                    unit=unit,
                    description=description
                )
    
    def _parse_data_section(self, content: str):
        """Parse ~A (ASCII Data) section."""
        data_match = re.search(r'~A.*', content, re.DOTALL)
        if not data_match:
            raise FileFormatError(
                "No ~A (Data) section found",
                suggestion="LAS file must contain data section"
            )
        
        data_text = data_match.group(0)
        
        # Remove ~A header
        data_lines = [line.strip() for line in data_text.split('\n')[1:] 
                     if line.strip() and not line.strip().startswith('#')]
        
        if not data_lines:
            raise DataValidationError(
                "No data found in LAS file",
                suggestion="Check that file contains actual log data"
            )
        
        # Parse data into array
        data_list = []
        for line in data_lines:
            try:
                values = [float(x) for x in line.split()]
                data_list.append(values)
            except ValueError:
                continue  # Skip malformed lines
        
        if not data_list:
            raise DataValidationError(
                "Could not parse any data rows",
                suggestion="Check data format in LAS file"
            )
        
        data_array = np.array(data_list)
        
        # Create DataFrame
        column_names = list(self.curves.keys())
        
        if data_array.shape[1] != len(column_names):
            raise DataValidationError(
                f"Data columns ({data_array.shape[1]}) don't match curve definitions ({len(column_names)})",
                suggestion="Check LAS file consistency"
            )
        
        self.data = pd.DataFrame(data_array, columns=column_names)
        
        # Replace null values with NaN
        self.data.replace(self.header.null_value, np.nan, inplace=True)
        
        # Set depth as index (usually first column)
        if len(column_names) > 0:
            depth_col = column_names[0]  # Typically DEPT or DEPTH
            self.data.set_index(depth_col, inplace=True)
    
    def _extract_parameter(self, text: str, param: str) -> Optional[str]:
        """Extract parameter value from LAS section."""
        pattern = rf'{param}\s*\.\s*(\S+)'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def get_curve_names(self) -> List[str]:
        """Get list of available curve names."""
        return list(self.curves.keys())
    
    def get_curve_data(self, curve_name: str) -> pd.Series:
        """Get data for specific curve."""
        if curve_name not in self.data.columns:
            raise DataValidationError(
                f"Curve '{curve_name}' not found",
                suggestion=f"Available curves: {', '.join(self.get_curve_names())}"
            )
        return self.data[curve_name]


class WellLogUpscaler:
    """
    Upscale well log data to reservoir grid cells.
    
    Averages high-resolution log data to match coarser grid resolution.
    """
    
    def __init__(self, method: str = 'arithmetic'):
        """
        Initialize upscaler.
        
        Args:
            method: Averaging method ('arithmetic', 'harmonic', 'geometric')
        """
        valid_methods = ['arithmetic', 'harmonic', 'geometric']
        if method not in valid_methods:
            raise DataValidationError(
                f"Invalid method: {method}",
                suggestion=f"Use one of: {', '.join(valid_methods)}"
            )
        self.method = method
    
    def upscale(
        self,
        depths: np.ndarray,
        values: np.ndarray,
        grid_tops: np.ndarray,
        grid_bottoms: np.ndarray,
    ) -> np.ndarray:
        """
        Upscale log data to grid cells.
        
        Args:
            depths: Log depths
            values: Log values
            grid_tops: Top depths of grid cells
            grid_bottoms: Bottom depths of grid cells
            
        Returns:
            Upscaled values for each grid cell
        """
        n_cells = len(grid_tops)
        upscaled = np.zeros(n_cells)
        
        for i in range(n_cells):
            top = grid_tops[i]
            bottom = grid_bottoms[i]
            
            # Find log values within this cell
            mask = (depths >= top) & (depths <= bottom)
            cell_values = values[mask]
            
            if len(cell_values) == 0:
                upscaled[i] = np.nan
                continue
            
            # Remove NaN values
            cell_values = cell_values[~np.isnan(cell_values)]
            
            if len(cell_values) == 0:
                upscaled[i] = np.nan
                continue
            
            # Apply averaging method
            if self.method == 'arithmetic':
                upscaled[i] = np.mean(cell_values)
            elif self.method == 'harmonic':
                # Harmonic mean (for permeability in series)
                upscaled[i] = len(cell_values) / np.sum(1.0 / (cell_values + 1e-10))
            elif self.method == 'geometric':
                # Geometric mean (for permeability)
                upscaled[i] = np.exp(np.mean(np.log(cell_values + 1e-10)))
        
        return upscaled


def load_las_file(filepath: str) -> Tuple[WellHeader, pd.DataFrame]:
    """
    Convenience function to load LAS file.
    
    Args:
        filepath: Path to LAS file
        
    Returns:
        Tuple of (header, data)
    """
    parser = LASParser(filepath)
    data = parser.parse()
    return parser.header, data


def upscale_well_logs(
    log_data: pd.DataFrame,
    grid_tops: np.ndarray,
    grid_bottoms: np.ndarray,
    curves: List[str],
    method: str = 'arithmetic',
) -> pd.DataFrame:
    """
    Upscale multiple well log curves to grid.
    
    Args:
        log_data: DataFrame with depth index and log curves
        grid_tops: Top depths of grid cells
        grid_bottoms: Bottom depths of grid cells
        curves: List of curve names to upscale
        method: Averaging method
        
    Returns:
        DataFrame with upscaled values
    """
    upscaler = WellLogUpscaler(method=method)
    
    depths = log_data.index.values
    upscaled_data = {}
    
    for curve in curves:
        if curve not in log_data.columns:
            continue
        
        values = log_data[curve].values
        upscaled_values = upscaler.upscale(depths, values, grid_tops, grid_bottoms)
        upscaled_data[curve] = upscaled_values
    
    result = pd.DataFrame(upscaled_data)
    result['top_depth'] = grid_tops
    result['bottom_depth'] = grid_bottoms
    
    return result
