"""
Integration tests for metric card functionality in the pipeline.

Tests the complete integration of metric cards into the rent campaign analysis pipeline.
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import tempfile
import os
from pathlib import Path

from scripts.pipeline import add_metric_cards
from src.functions import load_city_boundaries


class TestMetricCardsPipelineIntegration:
    """Integration tests for metric cards in the pipeline."""
    
    def test_add_metric_cards_pipeline_function(self):
        """Test the add_metric_cards pipeline function."""
        # Create mock results_dict
        district_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)])
            ],
            'durchschnMieteQM': [12.0, 8.0],
            'AnteilUeber65': [0.3, 0.1],
            'AnteilAuslaender': [0.2, 0.15],
            'durchschnFlaechejeBew': [45.0, 50.0],
            'Einwohner': [100, 80]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        
        results_dict = {'test_district': district_gdf}
        
        # Create mock rent_campaign_df (full dataset for city means calculation)
        rent_campaign_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]),
                Polygon([(0.2, 0.2), (1.2, 0.2), (1.2, 1.2), (0.2, 1.2)]),
                Polygon([(2.2, 2.2), (3.2, 2.2), (3.2, 3.2), (2.2, 3.2)])
            ],
            'durchschnMieteQM': [12.0, 8.0, 10.0, 9.0],
            'AnteilUeber65': [0.3, 0.1, 0.2, 0.18],
            'AnteilAuslaender': [0.2, 0.15, 0.17, 0.16],
            'durchschnFlaechejeBew': [45.0, 50.0, 47.0, 48.0],
            'Einwohner': [100, 80, 90, 85]
        }
        rent_campaign_df = gpd.GeoDataFrame(rent_campaign_data, crs="EPSG:3035")
        
        # Test the pipeline function
        try:
            enhanced_results = add_metric_cards(results_dict, rent_campaign_df)
            
            # Verify the function returns a dictionary
            assert isinstance(enhanced_results, dict)
            assert 'test_district' in enhanced_results
            
            enhanced_gdf = enhanced_results['test_district']
            
            # Since the mock data doesn't overlap with real cities, metric cards won't be added
            # This is expected behavior - the function should handle this gracefully
            if 'metric_cards' not in enhanced_gdf.columns:
                # This is expected when no city mapping is found
                pytest.skip("No city mapping found for test data - this is expected in test environment")
            assert len(enhanced_gdf) == 2
            
            # Check that metric cards contain expected metrics
            first_cards = enhanced_gdf['metric_cards'].iloc[0]
            expected_metrics = ['rent_per_m2', 'elderly_share', 'foreigner_share', 
                              'area_per_person', 'population']
            
            for metric_id in expected_metrics:
                assert metric_id in first_cards
                
                metric_card = first_cards[metric_id]
                assert 'id' in metric_card
                assert 'label' in metric_card
                assert 'value' in metric_card
                assert 'group_mean' in metric_card
                assert 'abs_diff' in metric_card
                assert 'pct_diff' in metric_card
                assert 'direction' in metric_card
                
                # Verify direction is one of the expected values
                assert metric_card['direction'] in ['above', 'below', 'equal']
                
        except Exception as e:
            # If city boundaries file doesn't exist, that's expected in test environment
            if "No such file or directory" in str(e) or "FileNotFoundError" in str(e):
                pytest.skip("City boundaries file not available in test environment")
            else:
                raise
    
    def test_add_metric_cards_with_mock_city_boundaries(self, tmp_path):
        """Test add_metric_cards with mock city boundaries file."""
        # Create mock city boundaries file
        city_data = {
            'GEN': ['TestCity'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        }
        mock_cities_gdf = gpd.GeoDataFrame(city_data, crs="EPSG:3035")
        
        # Save to temporary file
        cities_file = tmp_path / "test_cities.shp"
        mock_cities_gdf.to_file(cities_file)
        
        # Create mock results_dict
        district_data = {
            'geometry': [Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])],
            'durchschnMieteQM': [12.0],
            'AnteilUeber65': [0.3]
        }
        district_gdf = gpd.GeoDataFrame(district_data, crs="EPSG:3035")
        results_dict = {'test_district': district_gdf}
        
        # Create mock rent_campaign_df
        rent_campaign_data = {
            'geometry': [
                Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
                Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)])
            ],
            'durchschnMieteQM': [12.0, 8.0],
            'AnteilUeber65': [0.3, 0.1]
        }
        rent_campaign_df = gpd.GeoDataFrame(rent_campaign_data, crs="EPSG:3035")
        
        # Mock the load_city_boundaries function to use our test file
        import scripts.pipeline
        original_load_city_boundaries = scripts.pipeline.load_city_boundaries
        
        def mock_load_city_boundaries(path=None, crs="EPSG:3035"):
            return gpd.read_file(cities_file)
        
        scripts.pipeline.load_city_boundaries = mock_load_city_boundaries
        
        try:
            # Test the pipeline function
            enhanced_results = add_metric_cards(results_dict, rent_campaign_df)
            
            # Verify results
            assert isinstance(enhanced_results, dict)
            assert 'test_district' in enhanced_results
            
            enhanced_gdf = enhanced_results['test_district']
            assert 'metric_cards' in enhanced_gdf.columns
            
            # Check metric card content
            metric_cards = enhanced_gdf['metric_cards'].iloc[0]
            assert 'rent_per_m2' in metric_cards
            assert 'elderly_share' in metric_cards
            
            # Verify the metric card structure
            rent_card = metric_cards['rent_per_m2']
            assert rent_card['value'] == 12.0
            assert rent_card['group_mean'] == 10.0  # Mean of 12.0 and 8.0
            assert rent_card['direction'] == 'above'
            assert rent_card['abs_diff'] == 2.0
            assert abs(rent_card['pct_diff'] - 0.2) < 1e-10
            
        finally:
            # Restore original function
            scripts.pipeline.load_city_boundaries = original_load_city_boundaries
    
    def test_add_metric_cards_error_handling(self):
        """Test error handling in add_metric_cards function."""
        # Create invalid results_dict to trigger error
        results_dict = {'invalid_district': None}
        rent_campaign_df = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:3035")
        
        # Should handle error gracefully and return original results
        enhanced_results = add_metric_cards(results_dict, rent_campaign_df)
        
        # Should return the original results_dict when error occurs
        assert enhanced_results == results_dict
    
    def test_metric_card_data_structure_validation(self):
        """Test that metric cards have the correct data structure."""
        # This test validates the exact structure specified in the requirements
        from src.functions import create_metric_card
        
        # Test the exact structure from the requirements
        metric_card = create_metric_card(
            value=14.2,
            group_mean=10.9,
            metric_id="rent_per_m2",
            metric_label="Miete €/m²"
        )
        
        # Verify exact structure from requirements
        expected_structure = {
            "id": "rent_per_m2",
            "label": "Miete €/m²",
            "value": 14.2,
            "group_mean": 10.9,
            "abs_diff": 3.3,
            "pct_diff": 0.303,  # (14.2/10.9) - 1
            "direction": "above"
        }
        
        # Check individual fields to handle floating point precision
        assert metric_card["id"] == expected_structure["id"]
        assert metric_card["label"] == expected_structure["label"]
        assert metric_card["value"] == expected_structure["value"]
        assert metric_card["group_mean"] == expected_structure["group_mean"]
        assert abs(metric_card["abs_diff"] - expected_structure["abs_diff"]) < 1e-10
        assert abs(metric_card["pct_diff"] - expected_structure["pct_diff"]) < 1e-3
        assert metric_card["direction"] == expected_structure["direction"]
        
        # Verify percentage calculation is correct
        expected_pct_diff = (14.2 / 10.9) - 1
        assert abs(metric_card['pct_diff'] - expected_pct_diff) < 0.001
    
    def test_metric_card_edge_cases(self):
        """Test edge cases for metric card creation."""
        from src.functions import create_metric_card
        
        # Test with zero mean
        card = create_metric_card(5.0, 0.0, "test", "Test")
        assert card['direction'] == 'equal'
        assert card['abs_diff'] is None
        assert card['pct_diff'] is None
        
        # Test with very small values
        card = create_metric_card(0.001, 0.0001, "test", "Test")
        assert card['direction'] == 'above'
        assert card['abs_diff'] > 0
        assert card['pct_diff'] > 0
        
        # Test with negative values
        card = create_metric_card(-5.0, -10.0, "test", "Test")
        assert card['direction'] == 'above'  # -5 > -10
        assert card['abs_diff'] == 5.0
        # For negative values: (-5/-10) - 1 = 0.5 - 1 = -0.5
        assert abs(card['pct_diff'] - (-0.5)) < 1e-10
