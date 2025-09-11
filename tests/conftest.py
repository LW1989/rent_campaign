#!/usr/bin/env python3
"""
Shared pytest fixtures for rent campaign tests.

This module provides common test fixtures that can be used across all test modules.
"""

import tempfile
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_directory():
    """Create a temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_geometries():
    """Create sample polygon geometries for testing."""
    return [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    ]


@pytest.fixture
def sample_rent_gdf(sample_geometries):
    """Create a sample rent GeoDataFrame based on real data structure."""
    return gpd.GeoDataFrame({
        'durchschnMieteQM': [6.26, 11.67, 4.45, 12.92],  # Based on real rent values
        'AnzahlWohnungen': [3, 5, 3, 3],  # Based on real apartment counts
        'geometry': sample_geometries
    }, crs='EPSG:3035')


@pytest.fixture
def sample_renter_data(sample_geometries):
    """Create sample renter/ownership data based on real CSV format."""
    return gpd.GeoDataFrame({
        'Eigentuemerquote': [40.0, 35.5, 100.0, 67.2],  # Real values include 100.0 for KLAMMERN cases
        'geometry': sample_geometries
    }, crs='EPSG:4326')


@pytest.fixture
def sample_heating_type_data(sample_geometries):
    """Create sample heating type data."""
    return gpd.GeoDataFrame({
        'Fernheizung': [10, 5, 8, 12],
        'Etagenheizung': [20, 15, 12, 18],
        'Blockheizung': [5, 8, 6, 4],
        'Zentralheizung': [30, 25, 35, 28],  # High central heating
        'Einzel_Mehrraumoefen': [5, 7, 4, 6],
        'keine_Heizung': [0, 0, 0, 0],
        'geometry': sample_geometries
    }, crs='EPSG:4326')


@pytest.fixture
def sample_energy_type_data(sample_geometries):
    """Create sample energy type data."""
    return gpd.GeoDataFrame({
        'Gas': [40, 35, 30, 38],  # High fossil fuel usage
        'Heizoel': [20, 15, 25, 18],
        'Holz_Holzpellets': [5, 8, 6, 7],
        'Biomasse_Biogas': [2, 3, 2, 2],
        'Solar_Geothermie_Waermepumpen': [8, 12, 10, 9],
        'Strom': [10, 15, 12, 11],
        'Kohle': [5, 2, 8, 3],
        'Fernwaerme': [8, 8, 5, 10],  # Low district heating
        'kein_Energietraeger': [2, 2, 2, 2],
        'geometry': sample_geometries
    }, crs='EPSG:4326')


@pytest.fixture
def sample_threshold_dict():
    """Create sample threshold dictionary for testing."""
    return {
        "central_heating_thres": 0.4,
        "fossil_heating_thres": 0.4,
        "fernwaerme_thres": 0.1,
        "renter_share": 0.5
    }


@pytest.fixture
def wucher_detection_params():
    """Create sample Wucher detection parameters based on real params.py."""
    return {
        "method": "median",
        "threshold": 3.0,  # Match real params
        "neighborhood_size": 11,  # Match real params
        "min_rent_threshold": 6.0,  # Match real params
        "min_neighbors": 30,  # Match real params
        "rent_column": "durchschnMieteQM"
    }


@pytest.fixture
def outlier_test_grid():
    """Create a 5x5 grid with known outliers for testing outlier detection."""
    grid_size = 5
    rent_values = np.array([
        [8, 8, 8, 8, 8],      # Normal rents around 8 EUR/sqm
        [8, 8, 20, 8, 8],     # One outlier: 20 EUR/sqm
        [8, 8, 8, 8, 8],      # Normal rents
        [8, 25, 8, 8, 8],     # Another outlier: 25 EUR/sqm  
        [8, 8, 8, 8, 8]       # Normal rents
    ])
    
    # Create grid geometries (100m x 100m squares)
    geometries = []
    rent_flat = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Create 100m x 100m squares
            x_min = j * 100
            y_min = i * 100
            x_max = x_min + 100
            y_max = y_min + 100
            
            polygon = Polygon([
                (x_min, y_min), (x_max, y_min), 
                (x_max, y_max), (x_min, y_max)
            ])
            geometries.append(polygon)
            rent_flat.append(rent_values[i, j])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'durchschnMieteQM': rent_flat,
        'geometry': geometries
    }, crs='EPSG:3035')
    
    # Create xarray
    xarray = xr.DataArray(
        rent_values.astype(float),
        coords={'y': range(5), 'x': range(5)},
        dims=['y', 'x'],
        name='rent'
    )
    
    return {
        'gdf': gdf,
        'xarray': xarray,
        'outlier_positions': [(1, 2), (3, 1)],  # Known outlier positions
        'grid_size': grid_size
    }


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content based on real data structure."""
    return """GITTER_ID_100m;x_mp_100m;y_mp_100m;durchschnMieteQM;AnzahlWohnungen;werterlaeuternde_Zeichen
CRS3035RES100mN2691700E4341100;4341150;2691750;6,26;3;
CRS3035RES100mN2692400E4341200;4341250;2692450;11,67;5;
CRS3035RES100mN2694800E4343900;4343950;2694850;4,45;3;KLAMMERN
CRS3035RES100mN2696700E4341400;4341450;2696750;6,58;5;"""


@pytest.fixture
def sample_csv_file(temp_directory, sample_csv_content):
    """Create a temporary CSV file for testing."""
    csv_file = temp_directory / "test_data.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(sample_csv_content)
    return csv_file


@pytest.fixture
def large_test_dataset():
    """Create a larger dataset for performance testing."""
    def _create_large_dataset(size: int = 1000) -> gpd.GeoDataFrame:
        """Create a large test dataset with specified size."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate grid of squares
        grid_side = int(np.ceil(np.sqrt(size)))
        geometries = []
        rent_values = []
        
        base_rent = 10.0
        for i in range(grid_side):
            for j in range(grid_side):
                if len(geometries) >= size:
                    break
                    
                # Create 100m x 100m squares
                x_min = j * 100
                y_min = i * 100
                x_max = x_min + 100
                y_max = y_min + 100
                
                polygon = Polygon([
                    (x_min, y_min), (x_max, y_min), 
                    (x_max, y_max), (x_min, y_max)
                ])
                geometries.append(polygon)
                
                # Add some random variation and occasional outliers
                if np.random.random() < 0.05:  # 5% outliers
                    rent = base_rent + np.random.normal(15, 5)  # Outlier rents
                else:
                    rent = base_rent + np.random.normal(0, 2)  # Normal rents
                
                rent_values.append(max(1.0, rent))  # Ensure positive rents
        
        return gpd.GeoDataFrame({
            'durchschnMieteQM': rent_values[:size],
            'anzahl_wohnungen': np.random.randint(10, 200, size),
            'geometry': geometries[:size]
        }, crs='EPSG:3035')
    
    return _create_large_dataset


# Performance testing fixtures
@pytest.fixture
def benchmark_params():
    """Parameters for performance benchmarks."""
    return {
        'small_dataset_size': 100,
        'medium_dataset_size': 1000,
        'large_dataset_size': 10000,
        'timeout_seconds': 60
    }


# Parametrized fixtures for testing different scenarios
@pytest.fixture(params=[
    ('median', 2.0, 3),
    ('mean', 2.0, 3),
    ('median', 1.5, 5),
    ('median', 3.0, 3)
])
def outlier_detection_params(request):
    """Parametrized fixture for testing different outlier detection parameters."""
    method, threshold, neighborhood_size = request.param
    return {
        'method': method,
        'threshold': threshold,
        'neighborhood_size': neighborhood_size
    }


@pytest.fixture(params=[
    'EPSG:3035',  # Original CRS
    'EPSG:4326',  # WGS84
    'EPSG:3857'   # Web Mercator
])
def test_crs(request):
    """Parametrized fixture for testing different coordinate reference systems."""
    return request.param


# Real data fixtures for validation testing
@pytest.fixture
def real_data_paths():
    """Paths to real data files for validation testing."""
    from pathlib import Path
    base_path = Path(__file__).parent.parent
    
    return {
        'raw_csv_dir': base_path / 'data' / 'raw_csv',
        'processed_dir': base_path / 'data' / 'rent_campagne',
        'output_dir': base_path / 'output',
        'wucher_output': base_path / 'output' / 'wucher_miete' / 'wucher_miete_outliers.geojson'
    }


@pytest.fixture
def sample_real_rent_data(real_data_paths):
    """Load a small sample of real rent data for testing (if file exists)."""
    def _load_sample(n_rows=100):
        """Load n_rows from the real rent GeoJSON file."""
        try:
            import geopandas as gpd
            # Load only first n_rows for performance
            gdf = gpd.read_file(real_data_paths['processed_dir'] / 'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter.geojson')
            return gdf.head(n_rows) if len(gdf) > n_rows else gdf
        except FileNotFoundError:
            return None
    return _load_sample


@pytest.fixture
def real_csv_files(real_data_paths):
    """Get list of actual CSV files for validation testing."""
    csv_dir = real_data_paths['raw_csv_dir']
    if csv_dir.exists():
        return list(csv_dir.glob('*.csv'))
    return []


# Realistic GITTER_ID fixtures based on actual data
@pytest.fixture
def real_gitter_ids():
    """Real GITTER_ID examples from the actual dataset."""
    return [
        'CRS3035RES100mN2691700E4341100',  # Real ID from the data
        'CRS3035RES100mN2692400E4341200',  # Real ID from the data
        'CRS3035RES100mN2694800E4343900',  # Real ID from the data
        'CRS3035RES100mN2696700E4341400',  # Real ID from the data
    ]


@pytest.fixture
def realistic_rent_ranges():
    """Realistic rent value ranges based on actual data."""
    return {
        'normal_range': (4.0, 15.0),      # Most common rent range
        'low_range': (2.0, 6.0),          # Low rents
        'high_range': (15.0, 30.0),       # High rents (potential outliers)
        'outlier_range': (30.0, 100.0),   # Clear outliers
        'apartment_counts': (1, 50)        # Typical apartment count range
    }


@pytest.fixture
def real_coordinate_bounds():
    """Real coordinate bounds from the actual dataset (EPSG:3035)."""
    return {
        'x_min': 4341000,
        'x_max': 4350000,
        'y_min': 2691000,
        'y_max': 2700000
    }


# Memory and performance testing fixtures
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import gc
    from contextlib import contextmanager
    
    def _get_memory_mb():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @contextmanager
    def _memory_context():
        """Context manager for monitoring memory usage."""
        initial_memory = _get_memory_mb()
        gc.collect()  # Clean up before test
        yield _get_memory_mb
        final_memory = _get_memory_mb()
        memory_increase = final_memory - initial_memory
        return memory_increase
    
    return _memory_context
