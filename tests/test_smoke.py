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
import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.functions import (
    load_geojson_folder, get_rent_campaign_df, 
    calc_total, get_heating_type, get_energy_type, get_renter_share,
    gitter_id_to_polygon, convert_to_float, drop_cols, create_geodataframe,
    detect_neighbor_outliers, gdf_to_xarray, xarray_to_gdf, detect_wucher_miete
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

    def test_gitter_id_to_polygon(self):
        """Test GITTER_ID_100m to polygon conversion."""
        # Test valid GITTER_ID
        gitter_id = "CRS3035RES100mN2691700E4341100"
        polygon = gitter_id_to_polygon(gitter_id)
        
        self.assertIsNotNone(polygon)
        self.assertIsInstance(polygon, Polygon)
        
        # Check that it's a 100m x 100m square
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]  # max_x - min_x
        height = bounds[3] - bounds[1]  # max_y - min_y
        self.assertEqual(width, 100)
        self.assertEqual(height, 100)
        
        # Test invalid GITTER_ID
        invalid_id = "invalid_format"
        result = gitter_id_to_polygon(invalid_id)
        self.assertIsNone(result)

    def test_convert_to_float(self):
        """Test German decimal format conversion."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': ['CRS3035RES100mN2691700E4341100', 'CRS3035RES100mN2691800E4341200'],
            'value1': ['1,5', '2,3'],  # German decimal format
            'value2': ['10,25', '20,75'],
            'text_col': ['text1', 'text2']
        })
        
        result = convert_to_float(test_df)
        
        # Check that decimal conversion worked
        self.assertEqual(result['value1'].iloc[0], 1.5)
        self.assertEqual(result['value1'].iloc[1], 2.3)
        self.assertEqual(result['value2'].iloc[0], 10.25)
        self.assertEqual(result['value2'].iloc[1], 20.75)
        
        # Check that GITTER_ID column remained unchanged
        self.assertEqual(result['GITTER_ID_100m'].iloc[0], 'CRS3035RES100mN2691700E4341100')
        
        # Check that text columns were converted to numeric (should be NaN, then filled with 0)
        self.assertEqual(result['text_col'].iloc[0], 0)

    def test_drop_cols(self):
        """Test column dropping functionality."""
        test_df = pd.DataFrame({
            'keep1': [1, 2, 3],
            'keep2': [4, 5, 6],
            'drop1': [7, 8, 9],
            'drop2': [10, 11, 12],
            'nonexistent': [13, 14, 15]  # This column won't exist
        })
        
        # Drop some columns including one that doesn't exist
        result = drop_cols(test_df, ['drop1', 'drop2', 'nonexistent_col'])
        
        # Check that correct columns remain
        expected_cols = ['keep1', 'keep2', 'nonexistent']
        self.assertEqual(list(result.columns), expected_cols)
        self.assertEqual(len(result), 3)

    def test_create_geodataframe(self):
        """Test creation of GeoDataFrame from GITTER_ID."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': [
                'CRS3035RES100mN2691700E4341100',
                'CRS3035RES100mN2691800E4341200',
                'invalid_format'  # This should be filtered out
            ],
            'value': [1, 2, 3]
        })
        
        result = create_geodataframe(test_df)
        
        # Should be a GeoDataFrame
        self.assertIsInstance(result, gpd.GeoDataFrame)
        
        # Should have filtered out invalid GITTER_ID
        self.assertEqual(len(result), 2)
        
        # Should have proper CRS
        self.assertEqual(result.crs, "EPSG:3035")
        
        # Should have geometry column
        self.assertTrue('geometry' in result.columns)
        self.assertTrue(all(isinstance(geom, Polygon) for geom in result.geometry))


class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures for preprocessing tests."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV content
        self.test_csv_content = """GITTER_ID_100m;x_mp_100m;y_mp_100m;werterlaeuternde_Zeichen;value1;value2
CRS3035RES100mN2691700E4341100;4341150;2691750;-;1,5;10,25
CRS3035RES100mN2691800E4341200;4341250;2691850;-;2,3;20,75
CRS3035RES100mN2691900E4341300;4341350;2691950;-;3,1;30,50"""
        
        # Write test CSV file
        self.test_csv_path = Path(self.temp_dir) / "test_data.csv"
        with open(self.test_csv_path, 'w', encoding='utf-8') as f:
            f.write(self.test_csv_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preprocessing_workflow(self):
        """Test the complete preprocessing workflow with test data."""
        from src.functions import import_dfs, process_df
        
        # Test import_dfs
        df_dict = import_dfs(self.temp_dir, ";")
        
        self.assertEqual(len(df_dict), 1)
        self.assertIn('test_data', df_dict)
        
        test_df = df_dict['test_data']
        self.assertEqual(len(test_df), 3)
        self.assertIn('GITTER_ID_100m', test_df.columns)
        
        # Test full process_df workflow
        result_dict = process_df(
            path=self.temp_dir,
            sep=";",
            cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
            on_col="GITTER_ID_100m",
            drop_how="any",
            how="inner",
            gitter_id_column="GITTER_ID_100m"
        )
        
        self.assertEqual(len(result_dict), 1)
        result_gdf = result_dict['test_data']
        
        # Should be a GeoDataFrame
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        
        # Should have converted values correctly
        self.assertEqual(result_gdf['value1'].iloc[0], 1.5)
        self.assertEqual(result_gdf['value2'].iloc[0], 10.25)
        
        # Should have proper geometry
        self.assertTrue(all(isinstance(geom, Polygon) for geom in result_gdf.geometry))
        
        # Should have dropped specified columns
        dropped_cols = ["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen", "GITTER_ID_100m"]
        for col in dropped_cols:
            self.assertNotIn(col, result_gdf.columns)


class TestWucherDetection(unittest.TestCase):
    """Tests for Wucher Miete (rent gouging) detection functions."""
    
    def setUp(self):
        """Set up test fixtures for wucher detection tests."""
        # Create a simple 5x5 grid with known outliers
        self.grid_size = 5
        self.rent_values = np.array([
            [8, 8, 8, 8, 8],      # Normal rents around 8 EUR/sqm
            [8, 8, 20, 8, 8],     # One outlier: 20 EUR/sqm
            [8, 8, 8, 8, 8],      # Normal rents
            [8, 25, 8, 8, 8],     # Another outlier: 25 EUR/sqm  
            [8, 8, 8, 8, 8]       # Normal rents
        ])
        
        # Create grid geometries (100m x 100m squares)
        geometries = []
        rent_flat = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
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
                rent_flat.append(self.rent_values[i, j])
        
        # Create test GeoDataFrame
        self.test_gdf = gpd.GeoDataFrame({
            'durchschnMieteQM': rent_flat,
            'geometry': geometries
        }, crs='EPSG:3035')
        
        # Create test xarray
        self.test_xarray = xr.DataArray(
            self.rent_values.astype(float),  # Ensure float type for NaN handling
            coords={'y': range(5), 'x': range(5)},
            dims=['y', 'x'],
            name='rent'
        )
    
    def test_detect_neighbor_outliers_basic(self):
        """Test basic outlier detection with known data."""
        outliers = detect_neighbor_outliers(
            self.test_xarray, 
            method='median', 
            threshold=2.0, 
            size=3
        )
        
        # Should be boolean array
        self.assertEqual(outliers.dtype, bool)
        self.assertEqual(outliers.shape, (5, 5))
        
        # Should detect the outliers we placed
        outlier_positions = [(1, 2), (3, 1)]  # (row, col) of our outliers
        for row, col in outlier_positions:
            self.assertTrue(outliers.values[row, col], 
                          f"Failed to detect outlier at position ({row}, {col})")
        
        # Should not flag normal values as outliers
        normal_positions = [(0, 0), (2, 2), (4, 4)]
        for row, col in normal_positions:
            self.assertFalse(outliers.values[row, col],
                           f"Incorrectly flagged normal value at ({row}, {col}) as outlier")
    
    def test_detect_neighbor_outliers_parameters(self):
        """Test outlier detection with different parameters."""
        # Test with mean method
        outliers_mean = detect_neighbor_outliers(
            self.test_xarray, method='mean', threshold=2.0, size=3
        )
        self.assertEqual(outliers_mean.dtype, bool)
        
        # Test with different thresholds
        outliers_strict = detect_neighbor_outliers(
            self.test_xarray, threshold=5.0, size=3  # Very strict
        )
        outliers_loose = detect_neighbor_outliers(
            self.test_xarray, threshold=1.0, size=3  # Very loose
        )
        
        # Stricter threshold should find fewer outliers
        self.assertLessEqual(outliers_strict.sum(), outliers_loose.sum())
        
        # Test with different neighborhood sizes
        outliers_small = detect_neighbor_outliers(
            self.test_xarray, size=3, threshold=2.0
        )
        outliers_large = detect_neighbor_outliers(
            self.test_xarray, size=5, threshold=2.0
        )
        
        self.assertEqual(outliers_small.shape, outliers_large.shape)
    
    def test_detect_neighbor_outliers_edge_cases(self):
        """Test outlier detection edge cases."""
        # Test with NaN values
        test_array_nan = self.test_xarray.copy()
        test_array_nan.values[0, 0] = np.nan
        
        outliers_nan = detect_neighbor_outliers(test_array_nan, threshold=2.0)
        self.assertFalse(outliers_nan.values[0, 0])  # NaN should not be flagged
        
        # Test error cases
        with self.assertRaises(ValueError):
            detect_neighbor_outliers(self.test_xarray, size=2)  # Even size
        
        with self.assertRaises(ValueError):
            detect_neighbor_outliers(self.test_xarray, method='invalid')  # Invalid method
        
        # Test with 1D array (should fail)
        array_1d = xr.DataArray([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            detect_neighbor_outliers(array_1d)
    
    def test_gdf_to_xarray_conversion(self):
        """Test GeoDataFrame to xarray conversion."""
        result_xarray = gdf_to_xarray(self.test_gdf, 'durchschnMieteQM')
        
        # Check basic properties
        self.assertIsInstance(result_xarray, xr.DataArray)
        self.assertEqual(result_xarray.name, 'durchschnMieteQM')
        self.assertEqual(result_xarray.attrs['crs'], 'EPSG:3035')
        self.assertEqual(result_xarray.attrs['grid_size'], 100)
        
        # Check that values are preserved (allowing for some spatial tolerance)
        self.assertGreater(result_xarray.count(), 20)  # Should have most values
        
        # Check coordinate system
        self.assertIn('x', result_xarray.coords)
        self.assertIn('y', result_xarray.coords)
    
    def test_gdf_to_xarray_edge_cases(self):
        """Test GeoDataFrame to xarray edge cases."""
        # Test with empty GeoDataFrame
        empty_gdf = self.test_gdf.iloc[0:0]
        with self.assertRaises(ValueError):
            gdf_to_xarray(empty_gdf, 'durchschnMieteQM')
        
        # Test with missing column
        with self.assertRaises(ValueError):
            gdf_to_xarray(self.test_gdf, 'nonexistent_column')
        
        # Test with wrong CRS (should work with warning)
        gdf_wrong_crs = self.test_gdf.to_crs('EPSG:4326')
        result = gdf_to_xarray(gdf_wrong_crs, 'durchschnMieteQM')
        self.assertIsInstance(result, xr.DataArray)
    
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
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        self.assertIn('outlier_flag', result_gdf.columns)
        self.assertEqual(result_gdf['outlier_flag'].sum(), 2)  # Should have 2 True values
    
    def test_detect_wucher_miete_integration(self):
        """Test the main wucher detection function."""
        # Test with our synthetic data
        wucher_results = detect_wucher_miete(
            self.test_gdf,
            rent_column='durchschnMieteQM',
            threshold=1.5,  # Lower threshold to catch our test outliers
            neighborhood_size=3,
            min_rent_threshold=5.0  # Lower than our normal rents
        )
        
        # Should detect outliers
        self.assertIsInstance(wucher_results, gpd.GeoDataFrame)
        self.assertIn('wucher_miete_flag', wucher_results.columns)
        self.assertGreater(len(wucher_results), 0)  # Should find at least one outlier
        self.assertTrue(all(wucher_results['wucher_miete_flag']))  # All results should be flagged
        
        # Check that detected outliers have high rents
        detected_rents = wucher_results['durchschnMieteQM']
        self.assertGreater(detected_rents.min(), 15)  # Should be higher than normal rents
    
    def test_detect_wucher_miete_edge_cases(self):
        """Test wucher detection edge cases."""
        # Test with empty GeoDataFrame
        empty_gdf = self.test_gdf.iloc[0:0]
        with self.assertRaises(ValueError):
            detect_wucher_miete(empty_gdf)
        
        # Test with missing rent column
        with self.assertRaises(ValueError):
            detect_wucher_miete(self.test_gdf, rent_column='nonexistent')
        
        # Test with all rents below threshold
        high_threshold_result = detect_wucher_miete(
            self.test_gdf, 
            min_rent_threshold=100.0  # Higher than any rent in our data
        )
        self.assertEqual(len(high_threshold_result), 0)  # Should return empty
        
        # Test with invalid parameters
        with self.assertRaises(ValueError):
            detect_wucher_miete(self.test_gdf, neighborhood_size=4)  # Even size
        
        with self.assertRaises(ValueError):
            detect_wucher_miete(self.test_gdf, threshold=-1.0)  # Negative threshold
    
    def test_wucher_detection_parameters(self):
        """Test wucher detection with different parameter combinations."""
        # Test different methods
        results_median = detect_wucher_miete(
            self.test_gdf, method='median', threshold=1.5
        )
        results_mean = detect_wucher_miete(
            self.test_gdf, method='mean', threshold=1.5
        )
        
        # Both should find outliers
        self.assertGreater(len(results_median), 0)
        self.assertGreater(len(results_mean), 0)
        
        # Test different neighborhood sizes
        results_small = detect_wucher_miete(
            self.test_gdf, neighborhood_size=3, threshold=1.5
        )
        results_large = detect_wucher_miete(
            self.test_gdf, neighborhood_size=5, threshold=1.5
        )
        
        # Both should work
        self.assertIsInstance(results_small, gpd.GeoDataFrame)
        self.assertIsInstance(results_large, gpd.GeoDataFrame)


if __name__ == '__main__':
    unittest.main()
