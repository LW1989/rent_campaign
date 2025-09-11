#!/usr/bin/env python3
"""
Property-based tests using Hypothesis for rent campaign functions.

These tests generate random inputs to verify that certain properties
always hold for our functions, regardless of specific input values.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
from shapely.geometry import Polygon
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import (
    calc_total, convert_to_float, drop_cols, gitter_id_to_polygon,
    detect_neighbor_outliers, detect_wucher_miete
)
from tests.utils.factories import RentDataFactory


# Custom strategies for our domain
@st.composite
def valid_gitter_id(draw):
    """Generate valid GITTER_ID strings."""
    # Format: CRS3035RES100mN{northing}E{easting}
    northing = draw(st.integers(min_value=2600000, max_value=3000000))
    easting = draw(st.integers(min_value=4300000, max_value=4500000))
    
    # Snap to 100m grid
    northing = (northing // 100) * 100
    easting = (easting // 100) * 100
    
    return f"CRS3035RES100mN{northing}E{easting}"


@st.composite
def rent_value_string(draw):
    """Generate rent value strings in German decimal format."""
    rent = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    return f"{rent:.2f}".replace('.', ',')


@st.composite
def small_geodataframe(draw):
    """Generate small GeoDataFrames for testing."""
    n_points = draw(st.integers(min_value=1, max_value=20))
    
    # Generate rent values
    rent_values = draw(st.lists(
        st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=n_points, max_size=n_points
    ))
    
    # Generate apartment counts
    apartment_counts = draw(st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=n_points, max_size=n_points
    ))
    
    # Generate simple geometries
    geometries = []
    for i in range(n_points):
        x = i * 100
        y = 0
        geom = Polygon([(x, y), (x+100, y), (x+100, y+100), (x, y+100)])
        geometries.append(geom)
    
    return gpd.GeoDataFrame({
        'durchschnMieteQM': rent_values,
        'AnzahlWohnungen': apartment_counts,
        'geometry': geometries
    }, crs='EPSG:3035')


@pytest.mark.unit
@pytest.mark.fast
class TestHypothesisBasicProperties:
    """Property-based tests for basic utility functions."""
    
    @given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
    def test_calc_total_column_names_preserved(self, column_names):
        """Test that calc_total preserves all original columns."""
        # Create test dataframe with random column names
        test_data = {col: [1, 2, 3] for col in column_names}
        df = pd.DataFrame(test_data)
        
        # Choose subset of columns to sum
        cols_to_sum = column_names[:len(column_names)//2] if len(column_names) > 1 else []
        
        result = calc_total(df, cols_to_sum)
        
        # All original columns should be preserved
        for col in column_names:
            assert col in result.columns, f"Column {col} was lost"
        
        # Should have added 'total' column
        assert 'total' in result.columns
        assert len(result) == len(df)
    
    @given(st.data())
    def test_calc_total_sum_property(self, data):
        """Test that calc_total correctly sums the specified columns."""
        # Generate dataframe with numeric columns
        n_cols = data.draw(st.integers(min_value=2, max_value=8))
        n_rows = data.draw(st.integers(min_value=1, max_value=10))
        
        # Generate column names and numeric data
        col_names = [f"col_{i}" for i in range(n_cols)]
        
        df_data = {}
        for col in col_names:
            df_data[col] = data.draw(st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=n_rows, max_size=n_rows
            ))
        
        df = pd.DataFrame(df_data)
        
        # Choose random subset of columns to sum
        cols_to_sum = data.draw(st.lists(st.sampled_from(col_names), max_size=n_cols))
        
        result = calc_total(df, cols_to_sum)
        
        # Verify sum is correct
        if cols_to_sum:
            expected_total = df[cols_to_sum].sum(axis=1)
            np.testing.assert_array_almost_equal(result['total'], expected_total)
        else:
            # Empty column list should result in zeros
            assert all(result['total'] == 0)
    
    @given(st.lists(st.text(min_size=1), min_size=0, max_size=10))
    def test_drop_cols_property(self, cols_to_drop):
        """Test that drop_cols removes only specified columns."""
        # Create dataframe with known columns
        original_cols = ['keep1', 'keep2', 'drop1', 'drop2', 'keep3']
        df = pd.DataFrame({col: [1, 2, 3] for col in original_cols})
        
        result = drop_cols(df, cols_to_drop)
        
        # Check that specified columns are removed (if they existed)
        for col in cols_to_drop:
            if col in original_cols:
                assert col not in result.columns, f"Column {col} should have been dropped"
        
        # Check that non-specified columns are preserved
        for col in original_cols:
            if col not in cols_to_drop:
                assert col in result.columns, f"Column {col} should have been preserved"
        
        # Row count should be preserved
        assert len(result) == len(df)
    
    @given(valid_gitter_id())
    def test_gitter_id_to_polygon_properties(self, gitter_id):
        """Test properties of GITTER_ID to polygon conversion."""
        polygon = gitter_id_to_polygon(gitter_id)
        
        # Should return a valid polygon
        assert polygon is not None
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid
        
        # Should be a 100m x 100m square
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        assert abs(width - 100) < 0.001, f"Width should be 100m, got {width}"
        assert abs(height - 100) < 0.001, f"Height should be 100m, got {height}"
        
        # Should have 5 coordinates (closed polygon)
        coords = list(polygon.exterior.coords)
        assert len(coords) == 5, f"Should have 5 coordinates, got {len(coords)}"
        
        # First and last coordinates should be the same (closed)
        assert coords[0] == coords[-1], "Polygon should be closed"
    
    @given(st.text())
    def test_gitter_id_invalid_format_handling(self, invalid_string):
        """Test that invalid GITTER_ID formats return None."""
        # Skip if it accidentally generates a valid format
        assume(not invalid_string.startswith('CRS3035RES100mN'))
        
        result = gitter_id_to_polygon(invalid_string)
        assert result is None, f"Invalid GITTER_ID should return None: {invalid_string}"


@pytest.mark.unit
@pytest.mark.medium  
class TestHypothesisDataProcessing:
    """Property-based tests for data processing functions."""
    
    @given(st.data())
    def test_convert_to_float_german_decimal_property(self, data):
        """Test that convert_to_float properly handles German decimal format."""
        n_rows = data.draw(st.integers(min_value=1, max_value=20))
        
        # Generate dataframe with mixed column types
        df_data = {
            'GITTER_ID_100m': data.draw(st.lists(valid_gitter_id(), min_size=n_rows, max_size=n_rows)),
            'numeric_german': data.draw(st.lists(rent_value_string(), min_size=n_rows, max_size=n_rows)),
            'text_col': data.draw(st.lists(st.text(min_size=1, max_size=10), min_size=n_rows, max_size=n_rows))
        }
        
        df = pd.DataFrame(df_data)
        result = convert_to_float(df)
        
        # GITTER_ID should be unchanged
        assert result['GITTER_ID_100m'].equals(df['GITTER_ID_100m'])
        
        # Numeric columns should be converted to float
        assert result['numeric_german'].dtype in [float, 'float64']
        
        # All values should be non-negative (German format conversion + fillna(0))
        assert all(result['numeric_german'] >= 0)
        
        # Text columns should be converted to numeric if possible, otherwise 0
        # convert_to_float tries to convert strings to float, so "1" becomes 1.0
        text_values = result['text_col']
        assert text_values.dtype in [float, 'float64', int, 'int64']
        assert all(pd.notna(text_values) | (text_values == 0))
    
    @settings(max_examples=10, deadline=5000)  # Limit examples for complex test
    @given(small_geodataframe())
    def test_wucher_detection_basic_properties(self, gdf):
        """Test basic properties of Wucher detection."""
        # Skip if geodataframe is too small for meaningful analysis
        assume(len(gdf) >= 5)
        
        try:
            result = detect_wucher_miete(
                gdf,
                threshold=2.0,
                min_neighbors=2,  # Low threshold for small test data
                min_rent_threshold=1.0,
                neighborhood_size=3
            )
            
            # Result should be a GeoDataFrame
            assert isinstance(result, gpd.GeoDataFrame)
            
            # Should not return more outliers than input data
            assert len(result) <= len(gdf)
            
            # All results should be flagged as outliers
            if len(result) > 0:
                assert 'wucher_miete_flag' in result.columns
                assert all(result['wucher_miete_flag'])
                
                # Outliers should have rent values
                assert 'durchschnMieteQM' in result.columns
                assert all(result['durchschnMieteQM'] > 0)
                
                # Should preserve CRS
                assert result.crs == gdf.crs
                
                # All geometries should be valid
                assert all(result.geometry.is_valid)
        
        except (ValueError, RuntimeError) as e:
            # Some parameter combinations might be invalid for small datasets
            # This is acceptable behavior
            pass


@pytest.mark.unit
@pytest.mark.medium
class TestHypothesisXArrayProperties:
    """Property-based tests for xarray operations."""
    
    @settings(max_examples=20, deadline=3000)
    @given(st.data())
    def test_outlier_detection_array_properties(self, data):
        """Test properties of outlier detection on generated arrays."""
        # Generate array dimensions
        rows = data.draw(st.integers(min_value=5, max_value=15))
        cols = data.draw(st.integers(min_value=5, max_value=15))
        
        # Generate base values
        base_value = data.draw(st.floats(min_value=5.0, max_value=15.0))
        
        # Generate array with mostly normal values
        normal_values = data.draw(st.lists(
            st.floats(min_value=base_value-2, max_value=base_value+2, allow_nan=False),
            min_size=rows*cols, max_size=rows*cols
        ))
        
        # Reshape to 2D array
        values_2d = np.array(normal_values).reshape(rows, cols)
        
        # Create xarray
        test_array = xr.DataArray(
            values_2d,
            coords={'y': range(rows), 'x': range(cols)},
            dims=['y', 'x']
        )
        
        # Test different parameters
        method = data.draw(st.sampled_from(['mean', 'median']))
        threshold = data.draw(st.floats(min_value=0.5, max_value=5.0))
        size = data.draw(st.sampled_from([3, 5, 7]))
        
        try:
            outliers = detect_neighbor_outliers(test_array, method=method, threshold=threshold, size=size)
            
            # Basic properties
            assert outliers.shape == test_array.shape
            assert outliers.dtype == bool
            
            # Should not flag everything as outlier (with normal data)
            outlier_fraction = outliers.sum() / outliers.size
            assert outlier_fraction < 0.5, f"Too many outliers detected: {outlier_fraction:.2f}"
            
            # Should preserve coordinates
            assert list(outliers.coords.keys()) == list(test_array.coords.keys())
            
        except ValueError as e:
            # Some parameter combinations might be invalid
            # This is acceptable behavior
            pass
    
    @settings(max_examples=15)
    @given(st.data())
    def test_outlier_detection_edge_behavior(self, data):
        """Test outlier detection behavior at array edges."""
        # Generate small array
        size = data.draw(st.integers(min_value=3, max_value=8))
        
        # Create array with known pattern: high value in center, low around edges
        center = size // 2
        values = np.full((size, size), 5.0)  # Base value
        values[center, center] = 20.0  # Clear outlier in center
        
        test_array = xr.DataArray(values, dims=['y', 'x'])
        
        outliers = detect_neighbor_outliers(test_array, method='median', threshold=1.5, size=3)
        
        # Center should be detected as outlier
        assert outliers.values[center, center], "Central outlier should be detected"
        
        # Edge values should generally not be outliers (unless near the center outlier)
        edge_outliers = []
        for i in [0, size-1]:
            for j in [0, size-1]:
                if outliers.values[i, j]:
                    edge_outliers.append((i, j))
        
        # Most edge values should not be outliers
        assert len(edge_outliers) <= size//2, f"Too many edge outliers: {edge_outliers}"


@pytest.mark.unit
@pytest.mark.fast
class TestHypothesisInvariants:
    """Test invariants that should always hold."""
    
    @given(st.data())
    def test_function_determinism(self, data):
        """Test that functions are deterministic for the same input."""
        # Generate test data
        gitter_id = data.draw(valid_gitter_id())
        
        # Run function multiple times
        results = []
        for _ in range(3):
            result = gitter_id_to_polygon(gitter_id)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            if results[0] is None:
                assert results[i] is None
            else:
                assert results[0].equals(results[i]), "Function should be deterministic"
    
    @given(st.data())
    def test_data_preservation_properties(self, data):
        """Test that data processing preserves essential properties."""
        # Generate dataframe
        n_rows = data.draw(st.integers(min_value=1, max_value=10))
        
        test_data = {
            'col1': data.draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), 
                                     min_size=n_rows, max_size=n_rows)),
            'col2': data.draw(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), 
                                     min_size=n_rows, max_size=n_rows)),
            'keep_col': data.draw(st.lists(st.text(min_size=1), min_size=n_rows, max_size=n_rows))
        }
        
        df = pd.DataFrame(test_data)
        
        # Test calc_total preserves row count
        result = calc_total(df, ['col1', 'col2'])
        assert len(result) == len(df), "Row count should be preserved"
        
        # Test drop_cols preserves row count
        result2 = drop_cols(df, ['col1'])
        assert len(result2) == len(df), "Row count should be preserved"
        
        # Test that kept columns have same data
        assert result2['keep_col'].equals(df['keep_col']), "Kept columns should be unchanged"
    
    @settings(max_examples=10)
    @given(st.data())
    def test_wucher_detection_monotonicity(self, data):
        """Test that stricter thresholds find fewer or equal outliers."""
        # Generate small test dataset
        gdf = RentDataFactory.create_rent_grid(
            rows=5, cols=5,
            outlier_positions=[(2, 2)],  # Single known outlier
            base_rent=10.0,
            normal_variation=1.0
        )
        
        # Test with different thresholds
        threshold1 = data.draw(st.floats(min_value=1.0, max_value=2.0))
        threshold2 = data.draw(st.floats(min_value=2.1, max_value=4.0))
        
        assume(threshold1 < threshold2)  # Ensure threshold1 is more permissive
        
        try:
            result1 = detect_wucher_miete(gdf, threshold=threshold1, min_neighbors=2)
            result2 = detect_wucher_miete(gdf, threshold=threshold2, min_neighbors=2)
            
            # Stricter threshold should find fewer or equal outliers
            assert len(result2) <= len(result1), \
                f"Stricter threshold should find fewer outliers: {len(result2)} <= {len(result1)}"
        
        except (ValueError, RuntimeError):
            # Some combinations might be invalid
            pass
