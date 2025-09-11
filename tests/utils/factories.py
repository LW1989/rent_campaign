#!/usr/bin/env python3
"""
Test data factories for rent campaign tests.

This module provides factory functions to generate consistent, realistic test data
based on the actual data structure and value ranges.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shapely.geometry import Polygon, Point
from typing import List, Tuple, Dict, Optional, Union
import random


class RentDataFactory:
    """Factory for creating realistic rent data for testing."""
    
    @staticmethod
    def create_rent_grid(
        rows: int = 5,
        cols: int = 5,
        base_rent: float = 8.0,
        outlier_positions: Optional[List[Tuple[int, int]]] = None,
        outlier_rent_range: Tuple[float, float] = (20.0, 30.0),
        normal_variation: float = 2.0,
        grid_size_m: int = 100,
        crs: str = 'EPSG:3035',
        origin: Tuple[float, float] = (4341000, 2691000)
    ) -> gpd.GeoDataFrame:
        """
        Create a regular grid of rent data with controllable outliers.
        
        Args:
            rows: Number of grid rows
            cols: Number of grid columns 
            base_rent: Base rent value (EUR/sqm)
            outlier_positions: List of (row, col) positions for outliers
            outlier_rent_range: Min/max rent for outliers
            normal_variation: Standard deviation for normal rent variation
            grid_size_m: Size of each grid cell in meters
            crs: Coordinate reference system
            origin: Origin coordinates (x, y)
            
        Returns:
            GeoDataFrame with rent data and geometries
        """
        if outlier_positions is None:
            outlier_positions = []
            
        geometries = []
        rent_values = []
        apartment_counts = []
        
        x_origin, y_origin = origin
        
        for i in range(rows):
            for j in range(cols):
                # Create geometry
                x_min = x_origin + j * grid_size_m
                y_min = y_origin + i * grid_size_m
                x_max = x_min + grid_size_m
                y_max = y_min + grid_size_m
                
                polygon = Polygon([
                    (x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max)
                ])
                geometries.append(polygon)
                
                # Create rent value
                if (i, j) in outlier_positions:
                    rent = random.uniform(*outlier_rent_range)
                else:
                    rent = max(1.0, base_rent + random.gauss(0, normal_variation))
                
                rent_values.append(round(rent, 2))
                
                # Apartment count (realistic range 1-50)
                apartment_counts.append(random.randint(1, 20))
        
        return gpd.GeoDataFrame({
            'durchschnMieteQM': rent_values,
            'AnzahlWohnungen': apartment_counts,
            'geometry': geometries
        }, crs=crs)
    
    @staticmethod
    def create_realistic_rent_sample(
        n_points: int = 100,
        rent_distribution: str = 'mixed',
        outlier_fraction: float = 0.05,
        crs: str = 'EPSG:3035',
        bounds: Optional[Dict[str, float]] = None
    ) -> gpd.GeoDataFrame:
        """
        Create a realistic sample of rent data with various distributions.
        
        Args:
            n_points: Number of data points
            rent_distribution: 'low', 'normal', 'high', or 'mixed'
            outlier_fraction: Fraction of points that are outliers
            crs: Coordinate reference system
            bounds: Geographic bounds dict with x_min, x_max, y_min, y_max
            
        Returns:
            GeoDataFrame with realistic rent distribution
        """
        if bounds is None:
            bounds = {
                'x_min': 4341000, 'x_max': 4350000,
                'y_min': 2691000, 'y_max': 2700000
            }
        
        # Define rent ranges by distribution type
        rent_ranges = {
            'low': (2.0, 8.0),
            'normal': (6.0, 15.0),
            'high': (12.0, 25.0),
            'mixed': (2.0, 25.0)
        }
        
        rent_min, rent_max = rent_ranges[rent_distribution]
        n_outliers = int(n_points * outlier_fraction)
        n_normal = n_points - n_outliers
        
        # Generate rent values
        normal_rents = np.random.normal(
            loc=(rent_min + rent_max) / 2,
            scale=(rent_max - rent_min) / 6,  # 99.7% within range
            size=n_normal
        )
        normal_rents = np.clip(normal_rents, rent_min, rent_max)
        
        # Generate outlier rents
        outlier_rents = np.random.uniform(30.0, 60.0, size=n_outliers)
        
        # Combine and shuffle
        all_rents = np.concatenate([normal_rents, outlier_rents])
        np.random.shuffle(all_rents)
        
        # Generate random locations within bounds
        x_coords = np.random.uniform(bounds['x_min'], bounds['x_max'], n_points)
        y_coords = np.random.uniform(bounds['y_min'], bounds['y_max'], n_points)
        
        # Create geometries (100m x 100m squares)
        geometries = []
        for x, y in zip(x_coords, y_coords):
            # Snap to 100m grid
            x_snap = round(x / 100) * 100
            y_snap = round(y / 100) * 100
            
            polygon = Polygon([
                (x_snap, y_snap), (x_snap + 100, y_snap),
                (x_snap + 100, y_snap + 100), (x_snap, y_snap + 100)
            ])
            geometries.append(polygon)
        
        # Generate apartment counts
        apartment_counts = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
            size=n_points,
            p=[0.15, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03]
        )
        
        return gpd.GeoDataFrame({
            'durchschnMieteQM': np.round(all_rents, 2),
            'AnzahlWohnungen': apartment_counts,
            'geometry': geometries
        }, crs=crs)


class CSVDataFactory:
    """Factory for creating realistic CSV test data."""
    
    @staticmethod
    def create_rent_csv_content(
        n_rows: int = 10,
        include_special_cases: bool = True
    ) -> str:
        """
        Create realistic CSV content matching the real data format.
        
        Args:
            n_rows: Number of data rows to generate
            include_special_cases: Include KLAMMERN and empty cases
            
        Returns:
            CSV content as string
        """
        header = "GITTER_ID_100m;x_mp_100m;y_mp_100m;durchschnMieteQM;AnzahlWohnungen;werterlaeuternde_Zeichen\n"
        
        rows = []
        base_x, base_y = 4341000, 2691000
        
        for i in range(n_rows):
            # Generate realistic GITTER_ID
            x = base_x + (i % 10) * 100
            y = base_y + (i // 10) * 100
            x_center = x + 50
            y_center = y + 50
            
            gitter_id = f"CRS3035RES100mN{y}E{x}"
            
            # Generate rent (German decimal format with comma)
            if include_special_cases and i == n_rows - 1:
                # Add a KLAMMERN case
                rent_str = f"{random.uniform(4.0, 12.0):.2f}".replace('.', ',')
                apartments = random.randint(1, 10)
                special = "KLAMMERN"
            else:
                rent_str = f"{random.uniform(4.0, 20.0):.2f}".replace('.', ',')
                apartments = random.randint(1, 15)
                special = ""
            
            row = f"{gitter_id};{x_center};{y_center};{rent_str};{apartments};{special}"
            rows.append(row)
        
        return header + "\n".join(rows)
    
    @staticmethod
    def create_ownership_csv_content(
        n_rows: int = 10,
        include_missing: bool = True
    ) -> str:
        """
        Create realistic ownership CSV content.
        
        Args:
            n_rows: Number of data rows
            include_missing: Include missing values (–)
            
        Returns:
            CSV content as string
        """
        header = "GITTER_ID_100m;x_mp_100m;y_mp_100m;Eigentuemerquote;werterlaeuternde_Zeichen\n"
        
        rows = []
        base_x, base_y = 4341000, 2691000
        
        for i in range(n_rows):
            x = base_x + (i % 10) * 100
            y = base_y + (i // 10) * 100
            x_center = x + 50
            y_center = y + 50
            
            gitter_id = f"CRS3035RES100mN{y}E{x}"
            
            # Generate ownership percentage
            if include_missing and i % 4 == 0:
                ownership_str = "–"  # Missing value
                special = ""
            elif i % 7 == 0:
                ownership_str = "100,00"  # Full ownership
                special = "KLAMMERN"
            else:
                ownership_pct = random.uniform(20.0, 90.0)
                ownership_str = f"{ownership_pct:.2f}".replace('.', ',')
                special = ""
            
            row = f"{gitter_id};{x_center};{y_center};{ownership_str};{special}"
            rows.append(row)
        
        return header + "\n".join(rows)


class XArrayDataFactory:
    """Factory for creating xarray test data."""
    
    @staticmethod
    def create_rent_xarray(
        shape: Tuple[int, int] = (10, 10),
        base_rent: float = 8.0,
        outlier_positions: Optional[List[Tuple[int, int]]] = None,
        outlier_multiplier: float = 3.0,
        noise_level: float = 2.0,
        nan_fraction: float = 0.1
    ) -> xr.DataArray:
        """
        Create a 2D xarray with rent data for testing outlier detection.
        
        Args:
            shape: (rows, cols) shape of the array
            base_rent: Base rent value
            outlier_positions: Positions of outliers
            outlier_multiplier: Multiplier for outlier values
            noise_level: Standard deviation of noise
            nan_fraction: Fraction of values to set as NaN
            
        Returns:
            xarray DataArray with rent data
        """
        rows, cols = shape
        if outlier_positions is None:
            outlier_positions = []
        
        # Create base rent grid with noise
        rent_grid = np.random.normal(base_rent, noise_level, shape)
        
        # Add outliers
        for row, col in outlier_positions:
            if 0 <= row < rows and 0 <= col < cols:
                rent_grid[row, col] = base_rent * outlier_multiplier
        
        # Add some NaN values
        if nan_fraction > 0:
            n_nans = int(rows * cols * nan_fraction)
            nan_indices = np.random.choice(rows * cols, n_nans, replace=False)
            for idx in nan_indices:
                row, col = divmod(idx, cols)
                rent_grid[row, col] = np.nan
        
        # Ensure no negative rents
        rent_grid = np.maximum(rent_grid, 0.1)
        
        return xr.DataArray(
            rent_grid,
            coords={'y': range(rows), 'x': range(cols)},
            dims=['y', 'x'],
            name='durchschnMieteQM'
        )


# Convenience functions for common test scenarios
def create_outlier_test_scenario(
    scenario: str = 'sparse_outliers'
) -> Dict[str, Union[gpd.GeoDataFrame, xr.DataArray, List]]:
    """
    Create predefined test scenarios for outlier detection.
    
    Args:
        scenario: Type of scenario ('sparse_outliers', 'clustered_outliers', 'no_outliers')
        
    Returns:
        Dictionary with test data and expected results
    """
    scenarios = {
        'sparse_outliers': {
            'shape': (5, 5),
            'outlier_positions': [(1, 2), (3, 1)],
            'base_rent': 8.0,
            'expected_outliers': 2
        },
        'clustered_outliers': {
            'shape': (7, 7),
            'outlier_positions': [(2, 2), (2, 3), (3, 2), (3, 3)],
            'base_rent': 10.0,
            'expected_outliers': 4
        },
        'no_outliers': {
            'shape': (6, 6),
            'outlier_positions': [],
            'base_rent': 12.0,
            'expected_outliers': 0
        }
    }
    
    config = scenarios[scenario]
    
    # Create GeoDataFrame
    gdf = RentDataFactory.create_rent_grid(
        rows=config['shape'][0],
        cols=config['shape'][1],
        base_rent=config['base_rent'],
        outlier_positions=config['outlier_positions'],
        normal_variation=1.0  # Low variation for predictable tests
    )
    
    # Create xarray
    xarray = XArrayDataFactory.create_rent_xarray(
        shape=config['shape'],
        base_rent=config['base_rent'],
        outlier_positions=config['outlier_positions'],
        noise_level=1.0  # Low noise for predictable tests
    )
    
    return {
        'gdf': gdf,
        'xarray': xarray,
        'outlier_positions': config['outlier_positions'],
        'expected_outliers': config['expected_outliers'],
        'scenario_name': scenario
    }
