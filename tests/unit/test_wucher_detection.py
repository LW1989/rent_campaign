#!/usr/bin/env python3
"""
Unit tests for Wucher Miete (rent gouging) detection functions.

Migrated from unittest to pytest format with enhanced fixtures and parameterization.
"""

import pytest
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import (
    detect_neighbor_outliers, gdf_to_xarray, xarray_to_gdf, detect_wucher_miete
)
from tests.utils.factories import create_outlier_test_scenario


@pytest.mark.unit
@pytest.mark.fast
class TestNeighborOutlierDetection:
    """Test the core outlier detection algorithm."""
    
    def test_detect_neighbor_outliers_basic(self, outlier_test_grid):
        """Test basic outlier detection with known data."""
        outliers = detect_neighbor_outliers(
            outlier_test_grid['xarray'], 
            method='median', 
            threshold=2.0, 
            size=3
        )
        
        # Should be boolean array
        assert outliers.dtype == bool
        assert outliers.shape == outlier_test_grid['xarray'].shape
        
        # Should detect the outliers we placed
        outlier_positions = outlier_test_grid['outlier_positions']
        for row, col in outlier_positions:
            assert outliers.values[row, col], f"Failed to detect outlier at position ({row}, {col})"
        
        # Should not flag normal values as outliers
        normal_positions = [(0, 0), (2, 2), (4, 4)]
        for row, col in normal_positions:
            assert not outliers.values[row, col], f"Incorrectly flagged normal value at ({row}, {col}) as outlier"
    
    @pytest.mark.parametrize("method", ["median", "mean"])
    def test_detect_neighbor_outliers_methods(self, outlier_test_grid, method):
        """Test outlier detection with different statistical methods."""
        outliers = detect_neighbor_outliers(
            outlier_test_grid['xarray'], 
            method=method, 
            threshold=2.0, 
            size=3
        )
        
        assert outliers.dtype == bool
        assert outliers.shape == outlier_test_grid['xarray'].shape
        
        # Should detect at least some outliers with either method
        assert outliers.sum() > 0, f"No outliers detected with {method} method"
    
    @pytest.mark.parametrize("threshold,expected_range", [
        (1.0, (2, 8)),    # Loose threshold - more outliers (varies: 2-6)
        (2.0, (1, 4)),    # Medium threshold (varies: 1-3)
        (5.0, (0, 2)),    # Strict threshold - fewer outliers (varies: 0-2)
    ])
    def test_detect_neighbor_outliers_thresholds(self, outlier_test_grid, threshold, expected_range):
        """Test outlier detection with different threshold values."""
        outliers = detect_neighbor_outliers(
            outlier_test_grid['xarray'], 
            threshold=threshold, 
            size=3
        )
        
        outlier_count = outliers.sum().item()
        min_expected, max_expected = expected_range
        
        assert min_expected <= outlier_count <= max_expected, \
            f"Threshold {threshold} detected {outlier_count} outliers, expected {expected_range}"
    
    @pytest.mark.parametrize("neighborhood_size", [3, 5, 7])
    def test_detect_neighbor_outliers_sizes(self, outlier_test_grid, neighborhood_size):
        """Test outlier detection with different neighborhood sizes."""
        outliers = detect_neighbor_outliers(
            outlier_test_grid['xarray'], 
            size=neighborhood_size, 
            threshold=2.0
        )
        
        assert outliers.shape == outlier_test_grid['xarray'].shape
        assert outliers.dtype == bool
    
    def test_detect_neighbor_outliers_with_nans(self):
        """Test outlier detection with NaN values."""
        from tests.utils.factories import XArrayDataFactory
        
        test_array = XArrayDataFactory.create_rent_xarray(
            shape=(5, 5),
            outlier_positions=[(2, 2)],
            nan_fraction=0.2  # 20% NaN values
        )
        
        outliers = detect_neighbor_outliers(test_array, threshold=2.0)
        
        # NaN positions should not be flagged as outliers
        nan_mask = np.isnan(test_array.values)
        outliers_at_nan = outliers.values[nan_mask]
        
        assert not outliers_at_nan.any(), "NaN values were incorrectly flagged as outliers"
    
    def test_detect_neighbor_outliers_edge_cases(self):
        """Test outlier detection edge cases."""
        # Test with 1D array (should fail)
        array_1d = xr.DataArray([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="2D DataArrays"):
            detect_neighbor_outliers(array_1d)
        
        # Test with even size (should fail)
        test_array = xr.DataArray(np.ones((5, 5)), dims=['y', 'x'])
        with pytest.raises(ValueError, match="odd integer"):
            detect_neighbor_outliers(test_array, size=2)
        
        # Test with invalid method
        with pytest.raises(ValueError, match="method must be"):
            detect_neighbor_outliers(test_array, method='invalid')


@pytest.mark.unit
@pytest.mark.medium
class TestGeoDataFrameXArrayConversion:
    """Test conversion between GeoDataFrame and xarray formats."""
    
    def test_gdf_to_xarray_conversion(self, sample_rent_gdf):
        """Test GeoDataFrame to xarray conversion."""
        result_xarray = gdf_to_xarray(sample_rent_gdf, 'durchschnMieteQM')
        
        # Check basic properties
        assert isinstance(result_xarray, xr.DataArray)
        assert result_xarray.name == 'durchschnMieteQM'
        assert result_xarray.attrs['crs'] == 'EPSG:3035'
        assert result_xarray.attrs['grid_size'] == 100
        
        # Check that values are preserved (allowing for some spatial tolerance)
        assert result_xarray.count() > 0, "No values preserved in conversion"
        
        # Check coordinate system
        assert 'x' in result_xarray.coords
        assert 'y' in result_xarray.coords
    
    def test_gdf_to_xarray_edge_cases(self, sample_rent_gdf):
        """Test GeoDataFrame to xarray edge cases."""
        # Test with empty GeoDataFrame
        empty_gdf = sample_rent_gdf.iloc[0:0]
        with pytest.raises(ValueError):
            gdf_to_xarray(empty_gdf, 'durchschnMieteQM')
        
        # Test with missing column
        with pytest.raises(ValueError):
            gdf_to_xarray(sample_rent_gdf, 'nonexistent_column')
    
    def test_gdf_to_xarray_different_crs(self, sample_rent_gdf):
        """Test GeoDataFrame to xarray with different CRS."""
        # Convert to WGS84
        gdf_wrong_crs = sample_rent_gdf.to_crs('EPSG:4326')
        
        # Should work but with reprojection
        result = gdf_to_xarray(gdf_wrong_crs, 'durchschnMieteQM')
        assert isinstance(result, xr.DataArray)
    
    def test_xarray_to_gdf_conversion(self):
        """Test xarray to GeoDataFrame conversion."""
        # Create boolean outlier array
        outlier_array = xr.DataArray(
            np.array([[False, False, True], [False, True, False], [False, False, False]]),
            coords={'y': [0, 100, 200], 'x': [0, 100, 200]},
            dims=['y', 'x']
        )
        
        # Create corresponding template GeoDataFrame
        template_geometries = []
        for y in [50, 150, 250]:  # Centroids
            for x in [50, 150, 250]:
                polygon = Polygon([
                    (x-50, y-50), (x+50, y-50), 
                    (x+50, y+50), (x-50, y+50)
                ])
                template_geometries.append(polygon)
        
        template_gdf = gpd.GeoDataFrame({
            'dummy_col': range(9),
            'geometry': template_geometries
        }, crs='EPSG:3035')
        
        # Convert back
        result_gdf = xarray_to_gdf(outlier_array, template_gdf, 'outlier_flag')
        
        # Check results
        assert isinstance(result_gdf, gpd.GeoDataFrame)
        assert 'outlier_flag' in result_gdf.columns
        assert result_gdf['outlier_flag'].sum() == 2  # Should have 2 True values
    
    def test_xarray_to_gdf_mismatched_sizes(self, sample_rent_gdf):
        """Test xarray to GeoDataFrame with mismatched sizes."""
        # Create xarray with different size than template
        outlier_array = xr.DataArray(
            np.array([[True, False], [False, True]]),  # 2x2 array
            coords={'y': [0, 100], 'x': [0, 100]},
            dims=['y', 'x']
        )
        
        # Template has 4 geometries, but array only has 4 values - should work
        result_gdf = xarray_to_gdf(outlier_array, sample_rent_gdf, 'outlier_flag')
        assert isinstance(result_gdf, gpd.GeoDataFrame)


@pytest.mark.unit
@pytest.mark.medium
class TestWucherDetectionIntegration:
    """Test the main wucher detection function."""
    
    def test_detect_wucher_miete_basic(self, outlier_test_grid, wucher_detection_params):
        """Test the main wucher detection function with synthetic data."""
        # Use test-specific parameters
        test_params = wucher_detection_params.copy()
        test_params.update({
            'threshold': 1.5,  # Lower threshold to catch test outliers
            'neighborhood_size': 3,
            'min_rent_threshold': 5.0,  # Lower than test normal rents
            'min_neighbors': 2  # Lower for small test grid
        })
        
        wucher_results = detect_wucher_miete(
            outlier_test_grid['gdf'],
            **test_params
        )
        
        # Should detect outliers
        assert isinstance(wucher_results, gpd.GeoDataFrame)
        assert 'wucher_miete_flag' in wucher_results.columns
        assert len(wucher_results) > 0, "Should find at least one outlier"
        assert all(wucher_results['wucher_miete_flag']), "All results should be flagged"
        
        # Check that detected outliers have high rents
        detected_rents = wucher_results['durchschnMieteQM']
        assert detected_rents.min() > 15, "Should detect high-rent outliers"
    
    def test_detect_wucher_miete_no_outliers(self):
        """Test wucher detection with no outliers."""
        from tests.utils.factories import RentDataFactory
        
        # Create data with no outliers
        normal_data = RentDataFactory.create_rent_grid(
            rows=5, cols=5,
            base_rent=10.0,
            outlier_positions=[],  # No outliers
            normal_variation=1.0
        )
        
        result = detect_wucher_miete(
            normal_data,
            threshold=3.0,
            min_neighbors=5
        )
        
        # Should return empty result or very few outliers
        assert len(result) <= 2, "Should find few or no outliers in normal data"
    
    def test_detect_wucher_miete_parameter_validation(self, sample_rent_gdf):
        """Test parameter validation in wucher detection."""
        # Test with empty GeoDataFrame
        empty_gdf = sample_rent_gdf.iloc[0:0]
        with pytest.raises(ValueError):
            detect_wucher_miete(empty_gdf)
        
        # Test with missing rent column
        with pytest.raises(ValueError):
            detect_wucher_miete(sample_rent_gdf, rent_column='nonexistent')
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            detect_wucher_miete(sample_rent_gdf, neighborhood_size=4)  # Even size
        
        with pytest.raises(ValueError):
            detect_wucher_miete(sample_rent_gdf, threshold=-1.0)  # Negative threshold
    
    def test_detect_wucher_miete_high_threshold(self, sample_rent_gdf):
        """Test wucher detection with very high thresholds."""
        # Test with all rents below threshold
        high_threshold_result = detect_wucher_miete(
            sample_rent_gdf, 
            min_rent_threshold=100.0  # Higher than any rent in sample data
        )
        
        assert len(high_threshold_result) == 0, "Should return empty with high threshold"
    
    @pytest.mark.parametrize("method", ["median", "mean"])
    def test_detect_wucher_miete_different_methods(self, outlier_test_grid, method):
        """Test wucher detection with different statistical methods."""
        results = detect_wucher_miete(
            outlier_test_grid['gdf'], 
            method=method, 
            threshold=1.5,
            min_neighbors=2
        )
        
        # Both methods should find outliers
        assert len(results) > 0, f"No outliers found with {method} method"
        assert isinstance(results, gpd.GeoDataFrame)
    
    @pytest.mark.parametrize("neighborhood_size", [3, 5, 7])
    def test_detect_wucher_miete_neighborhood_sizes(self, outlier_test_grid, neighborhood_size):
        """Test wucher detection with different neighborhood sizes."""
        results = detect_wucher_miete(
            outlier_test_grid['gdf'], 
            neighborhood_size=neighborhood_size, 
            threshold=1.5,
            min_neighbors=2
        )
        
        # Should work with different neighborhood sizes
        assert isinstance(results, gpd.GeoDataFrame)
        # Different sizes might find different numbers of outliers, but should work
    
    def test_detect_wucher_miete_preserves_original_data(self, sample_rent_gdf):
        """Test that wucher detection doesn't modify original data."""
        original_data = sample_rent_gdf.copy()
        original_columns = set(sample_rent_gdf.columns)
        original_length = len(sample_rent_gdf)
        
        # Run detection
        detect_wucher_miete(sample_rent_gdf, min_neighbors=1)
        
        # Original data should be unchanged
        assert len(sample_rent_gdf) == original_length
        assert set(sample_rent_gdf.columns) == original_columns
        assert sample_rent_gdf.equals(original_data)
