"""
Unit tests for metric card functionality.

Tests the new metric card features that extend the rent campaign analysis
with JSON-like structures for demographic metrics.
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np

from src.functions import (
    load_city_boundaries, map_districts_to_cities, calculate_city_means,
    create_metric_card, add_metric_cards_to_districts
)


class TestLoadCityBoundaries:
    """Test city boundaries loading functionality."""
    
    def test_load_city_boundaries_success(self, tmp_path):
        """Test successful loading of city boundaries."""
        # Create a mock shapefile
        mock_data = {
            'GEN': ['Berlin', 'Hamburg', 'München'],
            'BEZ': ['Kreisfreie Stadt', 'Kreisfreie Stadt', 'Kreisfreie Stadt'],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
            ]
        }
        
        mock_gdf = gpd.GeoDataFrame(mock_data, crs="EPSG:3035")
        shapefile_path = tmp_path / "test_cities.shp"
        mock_gdf.to_file(shapefile_path)
        
        # Test loading
        result = load_city_boundaries(str(shapefile_path))
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['GEN', 'BEZ', 'geometry']
        assert result.crs == "EPSG:3035"
    
    def test_load_city_boundaries_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(Exception):
            load_city_boundaries("/nonexistent/path.shp")


class TestMapDistrictsToCities:
    """Test district to city mapping functionality."""
    
    def test_map_districts_to_cities_success(self):
        """Test successful mapping of districts to cities."""
        # Create mock city boundaries
        city_data = {
            'GEN': ['Berlin', 'Hamburg'],
            'BEZ': ['Kreisfreie Stadt', 'Kreisfreie Stadt'],
            'geometry': [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
            ]
        }
        krs_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        # Create mock district results
        district_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(3.5, 3.5), (4.5, 3.5), (4.5, 4.5), (3.5, 4.5)])
            ]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        
        results_dict = {
            'district_1': district_gdf.iloc[[0]],
            'district_2': district_gdf.iloc[[1]]
        }
        
        # Test mapping
        result = map_districts_to_cities(results_dict, krs_gdf)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'district_1' in result
        assert 'district_2' in result
        
        # Check that districts are mapped to correct cities
        assert result['district_1']['city_name'] == 'Berlin'
        assert result['district_2']['city_name'] == 'Hamburg'
    
    def test_map_districts_to_cities_empty_district(self):
        """Test handling of empty districts."""
        city_data = {
            'GEN': ['Berlin'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }
        krs_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        # Empty district
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:3035")
        results_dict = {'empty_district': empty_gdf}
        
        result = map_districts_to_cities(results_dict, krs_gdf)
        
        assert result['empty_district'] is None


class TestCalculateCityMeans:
    """Test city means calculation functionality."""
    
    def test_calculate_city_means_success(self):
        """Test successful calculation of city means."""
        # Create mock city boundaries
        city_data = {
            'GEN': ['Berlin', 'Hamburg'],
            'BEZ': ['Kreisfreie Stadt', 'Kreisfreie Stadt'],
            'geometry': [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
            ]
        }
        krs_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        # Create mock rent campaign data
        rent_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(0.7, 0.7), (1.7, 0.7), (1.7, 1.7), (0.7, 1.7)]),
                Polygon([(3.5, 3.5), (4.5, 3.5), (4.5, 4.5), (3.5, 4.5)])
            ],
            'durchschnMieteQM': [10.0, 12.0, 8.0],
            'AnteilUeber65': [0.2, 0.3, 0.15]
        }
        rent_campaign_df = gpd.GeoDataFrame(rent_data, crs="EPSG:3035")
        
        metric_columns = ['durchschnMieteQM', 'AnteilUeber65']
        
        # Test calculation
        result = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        assert isinstance(result, dict)
        assert 'Berlin' in result
        assert 'Hamburg' in result
        
        # Check Berlin means (should be average of 10.0 and 12.0)
        assert result['Berlin']['durchschnMieteQM'] == 11.0
        assert result['Berlin']['AnteilUeber65'] == 0.25
        
        # Check Hamburg means (should be 8.0)
        assert result['Hamburg']['durchschnMieteQM'] == 8.0
        assert result['Hamburg']['AnteilUeber65'] == 0.15
    
    def test_calculate_city_means_missing_columns(self):
        """Test handling of missing metric columns."""
        city_data = {
            'GEN': ['Berlin'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }
        krs_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        rent_data = {
            'geometry': [Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])],
            'durchschnMieteQM': [10.0]
        }
        rent_campaign_df = gpd.GeoDataFrame(rent_data, crs="EPSG:3035")
        
        metric_columns = ['durchschnMieteQM', 'missing_column']
        
        result = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        assert result['Berlin']['durchschnMieteQM'] == 10.0
        assert result['Berlin']['missing_column'] is None


class TestCreateMetricCard:
    """Test metric card creation functionality."""
    
    def test_create_metric_card_above_mean(self):
        """Test metric card creation when value is above mean."""
        result = create_metric_card(
            value=15.0,
            group_mean=10.0,
            metric_id="rent_per_m2",
            metric_label="Miete €/m²"
        )
        
        expected = {
            "id": "rent_per_m2",
            "label": "Miete €/m²",
            "value": 15.0,
            "group_mean": 10.0,
            "abs_diff": 5.0,
            "pct_diff": 0.5,
            "direction": "above"
        }
        
        assert result == expected
    
    def test_create_metric_card_below_mean(self):
        """Test metric card creation when value is below mean."""
        result = create_metric_card(
            value=8.0,
            group_mean=10.0,
            metric_id="rent_per_m2",
            metric_label="Miete €/m²"
        )
        
        expected = {
            "id": "rent_per_m2",
            "label": "Miete €/m²",
            "value": 8.0,
            "group_mean": 10.0,
            "abs_diff": -2.0,
            "pct_diff": -0.2,
            "direction": "below"
        }
        
        # Check individual fields to handle floating point precision
        assert result["id"] == expected["id"]
        assert result["label"] == expected["label"]
        assert result["value"] == expected["value"]
        assert result["group_mean"] == expected["group_mean"]
        assert result["abs_diff"] == expected["abs_diff"]
        assert abs(result["pct_diff"] - expected["pct_diff"]) < 1e-10
        assert result["direction"] == expected["direction"]
    
    def test_create_metric_card_equal_mean(self):
        """Test metric card creation when value equals mean."""
        result = create_metric_card(
            value=10.0,
            group_mean=10.0,
            metric_id="rent_per_m2",
            metric_label="Miete €/m²"
        )
        
        expected = {
            "id": "rent_per_m2",
            "label": "Miete €/m²",
            "value": 10.0,
            "group_mean": 10.0,
            "abs_diff": 0.0,
            "pct_diff": 0.0,
            "direction": "equal"
        }
        
        assert result == expected
    
    def test_create_metric_card_nan_values(self):
        """Test metric card creation with NaN values."""
        result = create_metric_card(
            value=np.nan,
            group_mean=10.0,
            metric_id="rent_per_m2",
            metric_label="Miete €/m²"
        )
        
        expected = {
            "id": "rent_per_m2",
            "label": "Miete €/m²",
            "value": np.nan,
            "group_mean": 10.0,
            "abs_diff": None,
            "pct_diff": None,
            "direction": "equal"
        }
        
        assert result == expected


class TestAddMetricCardsToDistricts:
    """Test adding metric cards to district results."""
    
    def test_add_metric_cards_success(self):
        """Test successful addition of metric cards."""
        # Create mock district data
        district_data = {
            'geometry': [Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])],
            'durchschnMieteQM': [12.0],
            'AnteilUeber65': [0.25]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        
        results_dict = {'test_district': district_gdf}
        
        # Mock city mapping
        district_city_mapping = {
            'test_district': {
                'city_name': 'Berlin',
                'city_type': 'Kreisfreie Stadt',
                'overlap_count': 1,
                'geometry': Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
            }
        }
        
        # Mock city means
        city_means = {
            'Berlin': {
                'durchschnMieteQM': 10.0,
                'AnteilUeber65': 0.2
            }
        }
        
        # Mock metric config
        metric_config = {
            'durchschnMieteQM': {
                'id': 'rent_per_m2',
                'label': 'Miete €/m²'
            },
            'AnteilUeber65': {
                'id': 'elderly_share',
                'label': 'Anteil über 65'
            }
        }
        
        # Test addition
        result = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, metric_config
        )
        
        assert isinstance(result, dict)
        assert 'test_district' in result
        
        enhanced_gdf = result['test_district']
        assert 'metric_cards' in enhanced_gdf.columns
        
        # Check metric cards content
        metric_cards = enhanced_gdf['metric_cards'].iloc[0]
        assert 'rent_per_m2' in metric_cards
        assert 'elderly_share' in metric_cards
        
        # Check rent metric card
        rent_card = metric_cards['rent_per_m2']
        assert rent_card['value'] == 12.0
        assert rent_card['group_mean'] == 10.0
        assert rent_card['direction'] == 'above'
        assert rent_card['abs_diff'] == 2.0
        assert abs(rent_card['pct_diff'] - 0.2) < 1e-10
    
    def test_add_metric_cards_empty_district(self):
        """Test handling of empty districts."""
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:3035")
        results_dict = {'empty_district': empty_gdf}
        
        district_city_mapping = {'empty_district': None}
        city_means = {}
        metric_config = {}
        
        result = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, metric_config
        )
        
        assert result['empty_district'].empty
        assert 'metric_cards' not in result['empty_district'].columns
    
    def test_add_metric_cards_no_city_mapping(self):
        """Test handling when no city mapping exists."""
        district_data = {
            'geometry': [Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])],
            'durchschnMieteQM': [12.0]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        
        results_dict = {'test_district': district_gdf}
        district_city_mapping = {'test_district': None}
        city_means = {}
        metric_config = {'durchschnMieteQM': {'id': 'rent_per_m2', 'label': 'Miete €/m²'}}
        
        result = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, metric_config
        )
        
        # Should return original data without metric cards
        assert result['test_district'].equals(district_gdf)
        assert 'metric_cards' not in result['test_district'].columns


class TestMetricCardIntegration:
    """Integration tests for the complete metric card workflow."""
    
    def test_complete_workflow(self):
        """Test the complete metric card workflow."""
        # Create comprehensive test data
        city_data = {
            'GEN': ['Berlin'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        }
        krs_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        # District data
        district_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)])
            ],
            'durchschnMieteQM': [12.0, 8.0],
            'AnteilUeber65': [0.3, 0.1]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        
        # Full rent campaign data (for city means calculation)
        rent_campaign_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]),
                Polygon([(0.2, 0.2), (1.2, 0.2), (1.2, 1.2), (0.2, 1.2)])
            ],
            'durchschnMieteQM': [12.0, 8.0, 10.0],
            'AnteilUeber65': [0.3, 0.1, 0.2]
        }
        rent_campaign_df = gpd.GeoDataFrame(rent_campaign_data, crs="EPSG:3035")
        
        results_dict = {'test_district': district_gdf}
        
        # Step 1: Map districts to cities
        district_city_mapping = map_districts_to_cities(results_dict, krs_gdf)
        
        # Step 2: Calculate city means
        metric_columns = ['durchschnMieteQM', 'AnteilUeber65']
        city_means = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        # Step 3: Add metric cards
        metric_config = {
            'durchschnMieteQM': {'id': 'rent_per_m2', 'label': 'Miete €/m²'},
            'AnteilUeber65': {'id': 'elderly_share', 'label': 'Anteil über 65'}
        }
        
        enhanced_results = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, metric_config
        )
        
        # Verify results
        assert len(enhanced_results) == 1
        assert 'test_district' in enhanced_results
        
        enhanced_gdf = enhanced_results['test_district']
        assert len(enhanced_gdf) == 2
        assert 'metric_cards' in enhanced_gdf.columns
        
        # Check first square's metric cards
        first_cards = enhanced_gdf['metric_cards'].iloc[0]
        assert 'rent_per_m2' in first_cards
        assert 'elderly_share' in first_cards
        
        # First square: rent=12.0, mean=10.0 -> above
        rent_card = first_cards['rent_per_m2']
        assert rent_card['value'] == 12.0
        assert rent_card['group_mean'] == 10.0
        assert rent_card['direction'] == 'above'
        
        # First square: elderly=0.3, mean=0.2 -> above
        elderly_card = first_cards['elderly_share']
        assert elderly_card['value'] == 0.3
        assert abs(elderly_card['group_mean'] - 0.2) < 1e-10
        assert elderly_card['direction'] == 'above'
        
        # Check second square's metric cards
        second_cards = enhanced_gdf['metric_cards'].iloc[1]
        
        # Second square: rent=8.0, mean=10.0 -> below
        rent_card = second_cards['rent_per_m2']
        assert rent_card['value'] == 8.0
        assert rent_card['group_mean'] == 10.0
        assert rent_card['direction'] == 'below'
        
        # Second square: elderly=0.1, mean=0.2 -> below
        elderly_card = second_cards['elderly_share']
        assert elderly_card['value'] == 0.1
        assert abs(elderly_card['group_mean'] - 0.2) < 1e-10
        assert elderly_card['direction'] == 'below'
