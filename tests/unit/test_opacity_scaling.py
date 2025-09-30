"""
Unit tests for population-based opacity scaling functionality.

Tests the new opacity scaling features:
- calculate_population_opacity()
- calculate_city_population_stats()
- map_squares_to_cities()
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from src.functions import (
    calculate_population_opacity,
    calculate_city_population_stats,
    map_squares_to_cities
)


@pytest.mark.unit
@pytest.mark.fast
class TestCalculatePopulationOpacity:
    """Test population opacity calculation helper function."""
    
    def test_opacity_min_value(self):
        """Test opacity for minimum population."""
        opacity = calculate_population_opacity(
            population=10,
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        assert opacity == 0.1
    
    def test_opacity_max_value(self):
        """Test opacity for maximum population."""
        opacity = calculate_population_opacity(
            population=100,
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        assert opacity == 0.6
    
    def test_opacity_middle_value(self):
        """Test opacity for median population."""
        opacity = calculate_population_opacity(
            population=55,
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # (55-10)/(100-10) = 0.5, so 0.1 + (0.5 * 0.5) = 0.35
        assert opacity == 0.35
    
    def test_opacity_equal_min_max(self):
        """Test opacity when all squares have same population."""
        opacity = calculate_population_opacity(
            population=50,
            city_min=50,
            city_max=50,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # Should return middle opacity
        assert opacity == 0.35  # (0.1 + 0.6) / 2
    
    def test_opacity_with_nan_population(self):
        """Test opacity with NaN population value."""
        opacity = calculate_population_opacity(
            population=np.nan,
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # Should return max_opacity as fallback
        assert opacity == 0.6
    
    def test_opacity_with_nan_city_stats(self):
        """Test opacity with NaN city statistics."""
        opacity = calculate_population_opacity(
            population=50,
            city_min=np.nan,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # Should return max_opacity as fallback
        assert opacity == 0.6
    
    def test_opacity_clipping_above_max(self):
        """Test that values above max are clipped."""
        opacity = calculate_population_opacity(
            population=500,  # Way above max
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # Should be clipped to max
        assert opacity == 0.6
    
    def test_opacity_clipping_below_min(self):
        """Test that values below min are clipped."""
        opacity = calculate_population_opacity(
            population=5,  # Below min
            city_min=10,
            city_max=100,
            min_opacity=0.1,
            max_opacity=0.6
        )
        # Should be clipped to min
        assert opacity == 0.1
    
    def test_opacity_custom_range(self):
        """Test opacity with custom opacity range."""
        opacity = calculate_population_opacity(
            population=55,
            city_min=10,
            city_max=100,
            min_opacity=0.2,
            max_opacity=0.9
        )
        # (55-10)/(100-10) = 0.5, so 0.2 + (0.5 * 0.7) = 0.55
        assert opacity == 0.55


@pytest.mark.unit
@pytest.mark.fast
class TestCalculateCityPopulationStats:
    """Test city population statistics calculation."""
    
    def test_basic_stats_calculation(self):
        """Test basic population statistics calculation."""
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [10, 50, 100, 200],
            'geometry': [
                Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
                Polygon([(0.1, 0), (0.2, 0), (0.2, 0.1), (0.1, 0.1)]),
                Polygon([(0, 0.1), (0.1, 0.1), (0.1, 0.2), (0, 0.2)]),
                Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)])
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'geometry': [Polygon([(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)])]
        }, crs='EPSG:4326')
        
        # Test with standard scaling (not robust)
        stats = calculate_city_population_stats(
            rent_df, city_gdf, 
            population_column='Einwohner',
            use_robust_scaling=False
        )
        
        assert 'Test City' in stats
        assert stats['Test City']['min'] == 10
        assert stats['Test City']['max'] == 200
        assert stats['Test City']['count'] == 4
    
    def test_robust_scaling_stats(self):
        """Test robust scaling with percentiles."""
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [2, 10, 15, 20, 25, 30, 35, 40, 500],  # 500 is outlier
            'geometry': [
                Polygon([(i*0.1, 0), ((i+1)*0.1, 0), ((i+1)*0.1, 0.1), (i*0.1, 0.1)])
                for i in range(9)
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])]
        }, crs='EPSG:4326')
        
        # Test with robust scaling
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner',
            use_robust_scaling=True,
            lower_percentile=5,
            upper_percentile=95
        )
        
        assert 'Test City' in stats
        # p95 should be less than the outlier (500)
        assert stats['Test City']['max'] < 500
        assert 'outliers_high' in stats['Test City']
        assert stats['Test City']['outliers_high'] >= 1
        assert stats['Test City']['actual_max'] == 500
    
    def test_multiple_cities(self):
        """Test statistics for multiple cities."""
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [10, 20, 100, 150],
            'geometry': [
                Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),      # City A
                Polygon([(0.1, 0), (0.2, 0), (0.2, 0.1), (0.1, 0.1)]),  # City A
                Polygon([(1, 0), (1.1, 0), (1.1, 0.1), (1, 0.1)]),      # City B
                Polygon([(1.1, 0), (1.2, 0), (1.2, 0.1), (1.1, 0.1)])   # City B
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['City A', 'City B'],
            'geometry': [
                Polygon([(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)]),
                Polygon([(1, 0), (1.3, 0), (1.3, 0.3), (1, 0.3)])
            ]
        }, crs='EPSG:4326')
        
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner',
            use_robust_scaling=False
        )
        
        assert 'City A' in stats
        assert 'City B' in stats
        assert stats['City A']['min'] == 10
        assert stats['City A']['max'] == 20
        assert stats['City B']['min'] == 100
        assert stats['City B']['max'] == 150
    
    def test_with_nan_values(self):
        """Test handling of NaN population values."""
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [10, np.nan, 100, 200],
            'geometry': [
                Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
                Polygon([(0.1, 0), (0.2, 0), (0.2, 0.1), (0.1, 0.1)]),
                Polygon([(0, 0.1), (0.1, 0.1), (0.1, 0.2), (0, 0.2)]),
                Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)])
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'geometry': [Polygon([(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)])]
        }, crs='EPSG:4326')
        
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner',
            use_robust_scaling=False
        )
        
        # NaN should be excluded from min/max
        assert stats['Test City']['min'] == 10
        assert stats['Test City']['max'] == 200
        assert stats['Test City']['count'] == 3  # One NaN excluded
    
    def test_single_square_city(self):
        """Test city with only one square."""
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [50],
            'geometry': [Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Small City'],
            'geometry': [Polygon([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)])]
        }, crs='EPSG:4326')
        
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner',
            use_robust_scaling=False
        )
        
        assert stats['Small City']['min'] == 50
        assert stats['Small City']['max'] == 50
        assert stats['Small City']['count'] == 1
    
    def test_missing_population_column(self):
        """Test handling of missing population column."""
        rent_df = gpd.GeoDataFrame({
            'other_column': [10, 20],
            'geometry': [
                Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
                Polygon([(0.1, 0), (0.2, 0), (0.2, 0.1), (0.1, 0.1)])
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'geometry': [Polygon([(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)])]
        }, crs='EPSG:4326')
        
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner'  # Column doesn't exist
        )
        
        # Should return empty dict
        assert stats == {}


@pytest.mark.unit
@pytest.mark.fast
class TestMapSquaresToCities:
    """Test square to city mapping functionality."""
    
    def test_basic_mapping(self):
        """Test basic mapping of squares to cities."""
        results_dict = {
            'test_district': gpd.GeoDataFrame({
                'GITTER_ID_100m': ['sq1', 'sq2'],
                'Einwohner': [50, 100],
                'geometry': [
                    Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
                    Polygon([(1, 0), (1.1, 0), (1.1, 0.1), (1, 0.1)])
                ]
            }, crs='EPSG:4326')
        }
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['City A', 'City B'],
            'geometry': [
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
                Polygon([(0.5, 0), (1.5, 0), (1.5, 0.5), (0.5, 0.5)])
            ]
        }, crs='EPSG:4326')
        
        mapping = map_squares_to_cities(results_dict, city_gdf)
        
        assert 'test_district' in mapping
        # Check that squares are mapped to cities
        assert len(mapping['test_district']) == 2
        # First square should be in City A, second in City B
        cities_mapped = set(mapping['test_district'].values())
        assert len(cities_mapped) >= 1  # At least one city mapped
    
    def test_empty_district(self):
        """Test mapping with empty district."""
        results_dict = {
            'empty_district': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        }
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['City A'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs='EPSG:4326')
        
        mapping = map_squares_to_cities(results_dict, city_gdf)
        
        assert 'empty_district' in mapping
        assert mapping['empty_district'] == {}
    
    def test_multiple_districts(self):
        """Test mapping with multiple districts."""
        results_dict = {
            'district_1': gpd.GeoDataFrame({
                'GITTER_ID_100m': ['sq1'],
                'geometry': [Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])]
            }, crs='EPSG:4326'),
            'district_2': gpd.GeoDataFrame({
                'GITTER_ID_100m': ['sq2'],
                'geometry': [Polygon([(1, 0), (1.1, 0), (1.1, 0.1), (1, 0.1)])]
            }, crs='EPSG:4326')
        }
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['City A', 'City B'],
            'geometry': [
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
                Polygon([(0.5, 0), (1.5, 0), (1.5, 0.5), (0.5, 0.5)])
            ]
        }, crs='EPSG:4326')
        
        mapping = map_squares_to_cities(results_dict, city_gdf)
        
        assert 'district_1' in mapping
        assert 'district_2' in mapping
        assert len(mapping['district_1']) == 1
        assert len(mapping['district_2']) == 1
    
    def test_square_outside_all_cities(self):
        """Test square that doesn't intersect any city."""
        results_dict = {
            'test_district': gpd.GeoDataFrame({
                'GITTER_ID_100m': ['sq1'],
                'geometry': [Polygon([(10, 10), (10.1, 10), (10.1, 10.1), (10, 10.1)])]
            }, crs='EPSG:4326')
        }
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['City A'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs='EPSG:4326')
        
        mapping = map_squares_to_cities(results_dict, city_gdf)
        
        assert 'test_district' in mapping
        # Square outside should have None or NaN as city (spatial join returns NaN for no match)
        values = list(mapping['test_district'].values())
        assert len(values) == 1
        # Check for None or NaN
        assert values[0] is None or pd.isna(values[0])


@pytest.mark.unit
@pytest.mark.fast
class TestOpacityScalingIntegration:
    """Integration tests for opacity scaling workflow."""
    
    def test_end_to_end_opacity_calculation(self):
        """Test complete opacity calculation workflow."""
        # Create test data with known population distribution
        rent_df = gpd.GeoDataFrame({
            'Einwohner': [10, 50, 100, 150],
            'geometry': [
                Polygon([(i*0.1, 0), ((i+1)*0.1, 0), ((i+1)*0.1, 0.1), (i*0.1, 0.1)])
                for i in range(4)
            ]
        }, crs='EPSG:4326')
        
        city_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'geometry': [Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])]
        }, crs='EPSG:4326')
        
        results_dict = {
            'test_district': rent_df.copy()
        }
        
        # Calculate stats
        stats = calculate_city_population_stats(
            rent_df, city_gdf,
            population_column='Einwohner',
            use_robust_scaling=False
        )
        
        # Map squares
        mapping = map_squares_to_cities(results_dict, city_gdf)
        
        # Get city stats for the test city
        city_stats = stats['Test City']
        
        # Calculate opacities for each population value
        opacities = []
        for pop in [10, 50, 100, 150]:
            opacity = calculate_population_opacity(
                pop, city_stats['min'], city_stats['max'], 0.1, 0.6
            )
            opacities.append(opacity)
        
        # Verify opacities are in ascending order (more population = higher opacity)
        assert opacities == sorted(opacities)
        
        # Verify first and last values
        assert opacities[0] == 0.1  # Min population → min opacity
        assert opacities[-1] == 0.6  # Max population → max opacity
