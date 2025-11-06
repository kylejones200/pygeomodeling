"""
Reservoir Engineering Module

Volumetrics, reserves calculation, and petrophysical relationships for
reservoir characterization and resource estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .exceptions import DataValidationError, InvalidParameterError


class ReservoirType(Enum):
    """Reservoir fluid type."""
    OIL = "oil"
    GAS = "gas"
    CONDENSATE = "condensate"


@dataclass
class VolumetricResult:
    """Container for volumetric calculations.
    
    Attributes:
        stoiip: Stock Tank Oil Initially In Place (STB) or GIIP (SCF)
        recoverable: Recoverable reserves
        gross_rock_volume: Gross rock volume (acre-ft or m³)
        net_rock_volume: Net rock volume
        pore_volume: Pore volume
        hydrocarbon_volume: Hydrocarbon pore volume
        recovery_factor: Recovery factor used
        reservoir_type: Oil or gas
    """
    stoiip: float
    recoverable: float
    gross_rock_volume: float
    net_rock_volume: float
    pore_volume: float
    hydrocarbon_volume: float
    recovery_factor: float
    reservoir_type: ReservoirType
    
    def __str__(self) -> str:
        unit = "STB" if self.reservoir_type == ReservoirType.OIL else "SCF"
        return (
            f"Volumetric Results ({self.reservoir_type.value.upper()}):\n"
            f"  STOIIP/GIIP: {self.stoiip:,.0f} {unit}\n"
            f"  Recoverable: {self.recoverable:,.0f} {unit}\n"
            f"  Recovery Factor: {self.recovery_factor:.1%}\n"
            f"  Gross Rock Volume: {self.gross_rock_volume:,.0f}\n"
            f"  Net Rock Volume: {self.net_rock_volume:,.0f}\n"
            f"  Pore Volume: {self.pore_volume:,.0f}\n"
            f"  HC Pore Volume: {self.hydrocarbon_volume:,.0f}"
        )


class VolumetricsCalculator:
    """
    Calculate reservoir volumetrics and reserves.
    
    Implements standard industry formulas for STOIIP/GIIP calculation.
    """
    
    def __init__(self, reservoir_type: ReservoirType = ReservoirType.OIL):
        """
        Initialize calculator.
        
        Args:
            reservoir_type: Oil or gas reservoir
        """
        self.reservoir_type = reservoir_type
    
    def calculate_stoiip(
        self,
        area: float,
        thickness: float,
        porosity: float,
        water_saturation: float,
        formation_volume_factor: float,
        unit_system: str = 'field',
    ) -> float:
        """
        Calculate Stock Tank Oil Initially In Place.
        
        Formula (Field units):
        STOIIP = 7758 × A × h × φ × (1 - Sw) / Bo
        
        Formula (Metric units):
        STOIIP = A × h × φ × (1 - Sw) / Bo
        
        Args:
            area: Reservoir area (acres or m²)
            thickness: Net pay thickness (ft or m)
            porosity: Average porosity (fraction)
            water_saturation: Average water saturation (fraction)
            formation_volume_factor: Oil formation volume factor (rb/stb or rm³/sm³)
            unit_system: 'field' (acres, ft) or 'metric' (m², m)
            
        Returns:
            STOIIP in STB or sm³
        """
        if not 0 <= porosity <= 1:
            raise DataValidationError(
                f"Porosity must be between 0 and 1, got {porosity}",
                suggestion="Use fraction (0.25) not percentage (25)"
            )
        
        if not 0 <= water_saturation <= 1:
            raise DataValidationError(
                f"Water saturation must be between 0 and 1, got {water_saturation}",
                suggestion="Use fraction not percentage"
            )
        
        hydrocarbon_saturation = 1 - water_saturation
        
        if unit_system == 'field':
            # 7758 converts acre-ft to barrels
            stoiip = 7758 * area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
        elif unit_system == 'metric':
            stoiip = area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
        else:
            raise InvalidParameterError(
                f"Unknown unit system: {unit_system}",
                valid_values=['field', 'metric']
            )
        
        return stoiip
    
    def calculate_giip(
        self,
        area: float,
        thickness: float,
        porosity: float,
        water_saturation: float,
        gas_expansion_factor: float,
        unit_system: str = 'field',
    ) -> float:
        """
        Calculate Gas Initially In Place.
        
        Formula (Field units):
        GIIP = 43560 × A × h × φ × (1 - Sw) / Bg
        
        Args:
            area: Reservoir area (acres or m²)
            thickness: Net pay thickness (ft or m)
            porosity: Average porosity (fraction)
            water_saturation: Average water saturation (fraction)
            gas_expansion_factor: Gas formation volume factor (rcf/scf or rm³/sm³)
            unit_system: 'field' or 'metric'
            
        Returns:
            GIIP in SCF or sm³
        """
        if not 0 <= porosity <= 1:
            raise DataValidationError(
                f"Porosity must be between 0 and 1, got {porosity}",
                suggestion="Use fraction not percentage"
            )
        
        if not 0 <= water_saturation <= 1:
            raise DataValidationError(
                f"Water saturation must be between 0 and 1, got {water_saturation}",
                suggestion="Use fraction not percentage"
            )
        
        hydrocarbon_saturation = 1 - water_saturation
        
        if unit_system == 'field':
            # 43560 converts acre-ft to cubic feet
            giip = 43560 * area * thickness * porosity * hydrocarbon_saturation / gas_expansion_factor
        elif unit_system == 'metric':
            giip = area * thickness * porosity * hydrocarbon_saturation / gas_expansion_factor
        else:
            raise InvalidParameterError(
                f"Unknown unit system: {unit_system}",
                valid_values=['field', 'metric']
            )
        
        return giip
    
    def calculate_from_grid(
        self,
        cell_volumes: np.ndarray,
        porosity: np.ndarray,
        water_saturation: np.ndarray,
        net_to_gross: Optional[np.ndarray] = None,
        formation_volume_factor: float = 1.2,
        recovery_factor: float = 0.35,
    ) -> VolumetricResult:
        """
        Calculate volumetrics from 3D grid properties.
        
        Args:
            cell_volumes: Cell volumes (ft³ or m³)
            porosity: Porosity per cell (fraction)
            water_saturation: Water saturation per cell (fraction)
            net_to_gross: Net-to-gross ratio per cell (optional)
            formation_volume_factor: Bo or Bg
            recovery_factor: Expected recovery factor
            
        Returns:
            VolumetricResult with all calculations
        """
        # Validate inputs
        if cell_volumes.shape != porosity.shape:
            raise DataValidationError(
                "Cell volumes and porosity must have same shape",
                suggestion="Check grid dimensions"
            )
        
        if porosity.shape != water_saturation.shape:
            raise DataValidationError(
                "Porosity and water saturation must have same shape",
                suggestion="Check grid dimensions"
            )
        
        # Apply net-to-gross if provided
        if net_to_gross is not None:
            if net_to_gross.shape != porosity.shape:
                raise DataValidationError(
                    "Net-to-gross must have same shape as other properties",
                    suggestion="Check grid dimensions"
                )
            effective_volumes = cell_volumes * net_to_gross
        else:
            effective_volumes = cell_volumes
        
        # Calculate volumes
        gross_rock_volume = np.sum(cell_volumes)
        net_rock_volume = np.sum(effective_volumes)
        
        # Pore volume
        pore_volume = np.sum(effective_volumes * porosity)
        
        # Hydrocarbon pore volume
        hydrocarbon_saturation = 1 - water_saturation
        hydrocarbon_volume = np.sum(effective_volumes * porosity * hydrocarbon_saturation)
        
        # STOIIP or GIIP
        if self.reservoir_type == ReservoirType.OIL:
            # Convert to stock tank barrels
            stoiip = hydrocarbon_volume / formation_volume_factor
        else:
            # Gas - already at standard conditions conceptually
            stoiip = hydrocarbon_volume / formation_volume_factor
        
        # Recoverable reserves
        recoverable = stoiip * recovery_factor
        
        return VolumetricResult(
            stoiip=stoiip,
            recoverable=recoverable,
            gross_rock_volume=gross_rock_volume,
            net_rock_volume=net_rock_volume,
            pore_volume=pore_volume,
            hydrocarbon_volume=hydrocarbon_volume,
            recovery_factor=recovery_factor,
            reservoir_type=self.reservoir_type
        )


class PetrophysicsCalculator:
    """
    Petrophysical relationships and transforms.
    
    Implements standard correlations used in reservoir characterization.
    """
    
    @staticmethod
    def archie_water_saturation(
        porosity: np.ndarray,
        resistivity: np.ndarray,
        water_resistivity: float,
        cementation_exponent: float = 2.0,
        saturation_exponent: float = 2.0,
        tortuosity_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Calculate water saturation using Archie's equation.
        
        Sw^n = (a / φ^m) × (Rw / Rt)
        
        Args:
            porosity: Porosity (fraction)
            resistivity: Formation resistivity (ohm-m)
            water_resistivity: Formation water resistivity (ohm-m)
            cementation_exponent: m (typically 1.8-2.0 for carbonates, 2.0-2.2 for sandstones)
            saturation_exponent: n (typically 2.0)
            tortuosity_factor: a (typically 0.62 for carbonates, 1.0 for sandstones)
            
        Returns:
            Water saturation (fraction)
        """
        # Formation factor: F = a / φ^m
        formation_factor = tortuosity_factor / (porosity ** cementation_exponent)
        
        # Resistivity index: RI = Rt / R0 = Rt / (F × Rw)
        # Sw^n = 1 / RI = (F × Rw) / Rt
        sw_n = (formation_factor * water_resistivity) / resistivity
        
        # Sw = (sw_n)^(1/n)
        water_saturation = sw_n ** (1.0 / saturation_exponent)
        
        # Clip to valid range
        water_saturation = np.clip(water_saturation, 0.0, 1.0)
        
        return water_saturation
    
    @staticmethod
    def kozeny_carman_permeability(
        porosity: np.ndarray,
        grain_size: float = 0.1,
        tortuosity: float = 1.5,
    ) -> np.ndarray:
        """
        Estimate permeability from porosity using Kozeny-Carman equation.
        
        k = (φ³ / (1-φ)²) × (d²) / (180 × τ²)
        
        Args:
            porosity: Porosity (fraction)
            grain_size: Average grain diameter (mm)
            tortuosity: Tortuosity factor (typically 1.5-3.0)
            
        Returns:
            Permeability (mD)
        """
        # Kozeny-Carman equation
        k = (porosity ** 3 / (1 - porosity + 1e-10) ** 2) * (grain_size ** 2) / (180 * tortuosity ** 2)
        
        # Convert to millidarcies (1 Darcy = 9.869233e-13 m²)
        # Simplified: multiply by large factor for mD
        k_md = k * 1e6  # Approximate conversion
        
        return k_md
    
    @staticmethod
    def timur_permeability(
        porosity: np.ndarray,
        water_saturation: np.ndarray,
        irreducible_water_saturation: float = 0.2,
    ) -> np.ndarray:
        """
        Estimate permeability using Timur correlation.
        
        k = 0.136 × (φ^4.4 / Swi^2)
        
        Args:
            porosity: Porosity (fraction)
            water_saturation: Water saturation (fraction)
            irreducible_water_saturation: Irreducible water saturation
            
        Returns:
            Permeability (mD)
        """
        # Use irreducible water saturation or current Sw, whichever is lower
        swi = np.minimum(water_saturation, irreducible_water_saturation)
        
        # Timur equation
        k = 0.136 * (porosity ** 4.4) / (swi ** 2 + 1e-10)
        
        return k
    
    @staticmethod
    def net_to_gross(
        porosity: np.ndarray,
        permeability: np.ndarray,
        water_saturation: np.ndarray,
        porosity_cutoff: float = 0.08,
        permeability_cutoff: float = 0.1,
        saturation_cutoff: float = 0.6,
    ) -> np.ndarray:
        """
        Calculate net-to-gross ratio based on cutoffs.
        
        Args:
            porosity: Porosity (fraction)
            permeability: Permeability (mD)
            water_saturation: Water saturation (fraction)
            porosity_cutoff: Minimum porosity for net pay
            permeability_cutoff: Minimum permeability for net pay (mD)
            saturation_cutoff: Maximum water saturation for net pay
            
        Returns:
            Net-to-gross ratio (0 or 1 per cell)
        """
        net_pay = (
            (porosity >= porosity_cutoff) &
            (permeability >= permeability_cutoff) &
            (water_saturation <= saturation_cutoff)
        )
        
        return net_pay.astype(float)
    
    @staticmethod
    def porosity_permeability_transform(
        porosity: np.ndarray,
        a: float = 1000.0,
        b: float = 10.0,
    ) -> np.ndarray:
        """
        Power-law porosity-permeability transform.
        
        k = a × φ^b
        
        Args:
            porosity: Porosity (fraction)
            a: Coefficient (typically 100-10000)
            b: Exponent (typically 3-10)
            
        Returns:
            Permeability (mD)
        """
        k = a * (porosity ** b)
        return k


def calculate_reserves_uncertainty(
    stoiip_p10: float,
    stoiip_p50: float,
    stoiip_p90: float,
    recovery_factor_p10: float,
    recovery_factor_p50: float,
    recovery_factor_p90: float,
) -> Dict[str, float]:
    """
    Calculate P10/P50/P90 reserves from STOIIP and recovery factor distributions.
    
    Args:
        stoiip_p10: STOIIP P10 (high estimate)
        stoiip_p50: STOIIP P50 (best estimate)
        stoiip_p90: STOIIP P90 (low estimate)
        recovery_factor_p10: RF P10
        recovery_factor_p50: RF P50
        recovery_factor_p90: RF P90
        
    Returns:
        Dictionary with P10/P50/P90 recoverable reserves
    """
    # Monte Carlo approach would be more rigorous, but for quick estimate:
    # Combine distributions assuming independence
    
    reserves_p10 = stoiip_p10 * recovery_factor_p10
    reserves_p50 = stoiip_p50 * recovery_factor_p50
    reserves_p90 = stoiip_p90 * recovery_factor_p90
    
    return {
        'P10': reserves_p10,  # High estimate (10% chance of exceeding)
        'P50': reserves_p50,  # Best estimate (50% chance)
        'P90': reserves_p90,  # Low estimate (90% chance of exceeding)
        'mean': (reserves_p10 + 4*reserves_p50 + reserves_p90) / 6,  # PERT approximation
    }


def decline_curve_analysis(
    time: np.ndarray,
    production_rate: np.ndarray,
    decline_type: str = 'exponential',
) -> Dict[str, float]:
    """
    Fit decline curve to production data.
    
    Args:
        time: Time array (days or months)
        production_rate: Production rate (bbl/day or Mcf/day)
        decline_type: 'exponential', 'hyperbolic', or 'harmonic'
        
    Returns:
        Dictionary with decline parameters and EUR
    """
    from scipy.optimize import curve_fit
    
    # Remove zeros and NaNs
    mask = (production_rate > 0) & ~np.isnan(production_rate)
    time_clean = time[mask]
    rate_clean = production_rate[mask]
    
    if len(time_clean) < 3:
        raise DataValidationError(
            "Need at least 3 valid data points for decline curve",
            suggestion="Provide more production history"
        )
    
    if decline_type == 'exponential':
        # q(t) = qi × exp(-D × t)
        def exponential_decline(t, qi, D):
            return qi * np.exp(-D * t)
        
        try:
            popt, _ = curve_fit(
                exponential_decline,
                time_clean,
                rate_clean,
                p0=[rate_clean[0], 0.001],
                bounds=([0, 0], [np.inf, 1])
            )
            qi, D = popt
            
            # EUR (to economic limit, assume 1 bbl/day)
            economic_limit = 1.0
            if D > 0:
                t_limit = -np.log(economic_limit / qi) / D
                eur = qi / D * (1 - np.exp(-D * t_limit))
            else:
                eur = np.inf
            
            return {
                'type': 'exponential',
                'qi': qi,
                'D': D,
                'EUR': eur,
                'economic_limit_time': t_limit if D > 0 else np.inf
            }
        
        except Exception as e:
            raise DataValidationError(
                f"Failed to fit exponential decline: {str(e)}",
                suggestion="Check data quality or try different decline type"
            )
    
    else:
        raise InvalidParameterError(
            f"Decline type '{decline_type}' not yet implemented",
            valid_values=['exponential']
        )
