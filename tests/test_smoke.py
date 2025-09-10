#!/usr/bin/env python3
"""
Smoke tests for rent campaign pipeline.

These tests verify that the pipeline can run end-to-end with minimal test data.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.functions import (
    load_geojson_folder, get_rent_campaign_df, 
    calc_total, get_heating_type, get_energy_type, get_renter_share
)


class TestSmokePipeline(unittest.TestCase):
    """Smoke tests for the rent campaign pipeline."""
    
    def setUp(self):
        """Set up test fixtures with minimal test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal test geodataframes
        self.test_geometry = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
        ]
        
        # Create test renter data
        self.renter_data = gpd.GeoDataFrame({
            'Eigentuemerquote': [40, 35, 30],  # Will be converted to renter share
            'geometry': self.test_geometry
        }, crs='EPSG:4326')
        
        # Create test heating type data
        self.heating_type_data = gpd.GeoDataFrame({
            'Fernheizung': [10, 5, 8],
            'Etagenheizung': [20, 15, 12],
            'Blockheizung': [5, 8, 6],
            'Zentralheizung': [30, 25, 35],  # High central heating
            'Einzel_Mehrraumoefen': [5, 7, 4],
            'keine_Heizung': [0, 0, 0],
            'geometry': self.test_geometry
        }, crs='EPSG:4326')
        
        # Create test energy type data  
        self.energy_type_data = gpd.GeoDataFrame({
            'Gas': [40, 35, 30],  # High fossil fuel usage
            'Heizoel': [20, 15, 25],
            'Holz_Holzpellets': [5, 8, 6],
            'Biomasse_Biogas': [2, 3, 2],
            'Solar_Geothermie_Waermepumpen': [8, 12, 10],
            'Strom': [10, 15, 12],
            'Kohle': [5, 2, 8],
            'Fernwaerme': [8, 8, 5],  # Low district heating
            'kein_Energietraeger': [2, 2, 2],
            'geometry': self.test_geometry
        }, crs='EPSG:4326')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calc_total_function(self):
        """Test the calc_total helper function."""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        result = calc_total(test_df, ['col1', 'col2'])
        
        self.assertIn('total', result.columns)
        self.assertEqual(result['total'].tolist(), [5, 7, 9])
        
    def test_get_heating_type(self):
        """Test heating type processing."""
        result = get_heating_type(self.heating_type_data.copy())
        
        self.assertIn('central_heating_share', result.columns)
        self.assertIn('geometry', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check that shares are calculated correctly
        self.assertTrue(all(0 <= share <= 1 for share in result['central_heating_share']))
    
    def test_get_energy_type(self):
        """Test energy type processing."""
        result = get_energy_type(self.energy_type_data.copy())
        
        self.assertIn('fossil_heating_share', result.columns)
        self.assertIn('fernwaerme_share', result.columns)
        self.assertIn('geometry', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check that shares are valid percentages
        self.assertTrue(all(0 <= share <= 1 for share in result['fossil_heating_share']))
        self.assertTrue(all(0 <= share <= 1 for share in result['fernwaerme_share']))
    
    def test_get_renter_share(self):
        """Test renter share processing."""
        result = get_renter_share(self.renter_data.copy())
        
        self.assertIn('renter_share', result.columns)
        self.assertIn('geometry', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check conversion from ownership to renter percentage
        expected_renter_shares = [0.60, 0.65, 0.70]  # 100 - [40, 35, 30] / 100
        self.assertEqual(result['renter_share'].tolist(), expected_renter_shares)
    
    def test_get_rent_campaign_df_integration(self):
        """Test the main rent campaign data processing function."""
        threshold_dict = {
            "central_heating_thres": 0.4,  # Lower threshold for test data
            "fossil_heating_thres": 0.4,
            "fernwaerme_thres": 0.1,
            "renter_share": 0.5
        }
        
        result = get_rent_campaign_df(
            heating_type=self.heating_type_data.copy(),
            energy_type=self.energy_type_data.copy(),
            renter_df=self.renter_data.copy(),
            threshold_dict=threshold_dict
        )
        
        # Check that the result has the expected columns
        expected_columns = [
            'geometry', 'central_heating_flag', 'fossil_heating_flag', 
            'fernwaerme_flag', 'renter_flag'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that flags are boolean
        for flag_col in [c for c in expected_columns if c.endswith('_flag')]:
            self.assertTrue(result[flag_col].dtype == bool)
        
        # Result should only contain areas where renter_flag is True
        self.assertTrue(all(result['renter_flag'] == True))
    
    def test_create_test_geojson_file(self):
        """Test creation and loading of a simple GeoJSON file."""
        # Create a test GeoJSON file
        test_file = Path(self.temp_dir) / "test.geojson"
        
        test_gdf = gpd.GeoDataFrame({
            'name': ['test1', 'test2'],
            'value': [1, 2],
            'geometry': self.test_geometry[:2]
        }, crs='EPSG:4326')
        
        test_gdf.to_file(test_file, driver='GeoJSON')
        
        # Test that the file was created
        self.assertTrue(test_file.exists())
        
        # Test loading with our function
        loaded_dict = load_geojson_folder(self.temp_dir)
        
        self.assertIn('test', loaded_dict)
        self.assertEqual(len(loaded_dict['test']), 2)
        self.assertIn('name', loaded_dict['test'].columns)
    
    def test_pipeline_minimal_data(self):
        """Test that the pipeline can process minimal valid data without errors."""
        # This is a minimal integration test
        threshold_dict = {
            "central_heating_thres": 0.3,
            "fossil_heating_thres": 0.3,
            "fernwaerme_thres": 0.1,
            "renter_share": 0.5
        }
        
        # Process the data through the pipeline
        result = get_rent_campaign_df(
            heating_type=self.heating_type_data.copy(),
            energy_type=self.energy_type_data.copy(),
            renter_df=self.renter_data.copy(),
            threshold_dict=threshold_dict
        )
        
        # Basic validation that pipeline runs and produces output
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertGreater(len(result), 0)  # Should have some results
        self.assertTrue(hasattr(result, 'geometry'))
        self.assertEqual(result.crs, self.renter_data.crs)


if __name__ == '__main__':
    unittest.main()
