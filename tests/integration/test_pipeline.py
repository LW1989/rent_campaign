#!/usr/bin/env python3
"""
Integration tests for the complete rent campaign pipeline.

These tests verify that different pipeline components work together correctly.
"""

import pytest
import geopandas as gpd
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import (
    get_rent_campaign_df, process_df, detect_wucher_miete,
    load_geojson_folder
)


@pytest.mark.integration
@pytest.mark.medium
class TestPipelineIntegration:
    """Test integration between pipeline components."""
    
    def test_get_rent_campaign_df_integration(
        self, 
        sample_heating_type_data, 
        sample_energy_type_data, 
        sample_renter_data,
        sample_threshold_dict
    ):
        """Test the main rent campaign data processing function."""
        # Define the column lists and labels as in the prototype
        heating_typeshare_list = [
            "Fernheizung_share",
            "Etagenheizung_share", 
            "Blockheizung_share",
            "Zentralheizung_share",
            "Einzel_Mehrraumoefen_share",
            "keine_Heizung_share",
        ]
        
        energy_type_share_list = [
            "fossil_heating_share",
            "renewable_share", 
            "no_energy_type",
        ]
        
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung", 
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Öfen",
            "keine_Heizung_share": "Keine Heizung",
        }
        
        energy_labels = {
            "fossil_heating_share": "Fossil",
            "renewable_share": "Erneuerbar",
            "no_energy_type": "Keine Angabe",
        }
        
        result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels,
            threshold_dict=sample_threshold_dict
        )
        
        # Check that the result has the expected columns
        expected_columns = [
            'geometry', 'central_heating_flag', 'fossil_heating_flag', 
            'fernwaerme_flag', 'renter_flag', 'heating_pie', 'energy_pie'
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check that flags are boolean
        flag_columns = [c for c in expected_columns if c.endswith('_flag')]
        for flag_col in flag_columns:
            assert result[flag_col].dtype == bool, f"Column {flag_col} is not boolean"
        
        # Check that pie columns contain lists of dictionaries
        assert all(isinstance(x, list) for x in result['heating_pie']), "heating_pie column should contain lists"
        assert all(isinstance(x, list) for x in result['energy_pie']), "energy_pie column should contain lists"
        
        # Check that each pie entry is a list of dictionaries with label and value
        for pie_list in result['heating_pie']:
            assert all(isinstance(item, dict) for item in pie_list), "heating_pie items should be dictionaries"
            for item in pie_list:
                assert 'label' in item, "heating_pie items should have 'label' key"
                assert 'value' in item, "heating_pie items should have 'value' key"
                assert isinstance(item['value'], (int, float)), "heating_pie values should be numeric"
        
        for pie_list in result['energy_pie']:
            assert all(isinstance(item, dict) for item in pie_list), "energy_pie items should be dictionaries"
            for item in pie_list:
                assert 'label' in item, "energy_pie items should have 'label' key"
                assert 'value' in item, "energy_pie items should have 'value' key"
                assert isinstance(item['value'], (int, float)), "energy_pie values should be numeric"
        
        # Result should only contain areas where renter_flag is True
        assert all(result['renter_flag'] == True), "Non-renter areas should be filtered out"
        
        # Should be a valid GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs is not None
    
    def test_pipeline_with_different_thresholds(
        self,
        sample_heating_type_data,
        sample_energy_type_data, 
        sample_renter_data
    ):
        """Test pipeline with different threshold configurations."""
        # Strict thresholds
        strict_thresholds = {
            "central_heating_thres": 0.8,
            "fossil_heating_thres": 0.8,
            "fernwaerme_thres": 0.1,
            "renter_share": 0.8
        }
        
        # Loose thresholds  
        loose_thresholds = {
            "central_heating_thres": 0.2,
            "fossil_heating_thres": 0.2,
            "fernwaerme_thres": 0.1,
            "renter_share": 0.2
        }
        
        # Define the column lists and labels as in the prototype
        heating_typeshare_list = [
            "Fernheizung_share",
            "Etagenheizung_share", 
            "Blockheizung_share",
            "Zentralheizung_share",
            "Einzel_Mehrraumoefen_share",
            "keine_Heizung_share",
        ]
        
        energy_type_share_list = [
            "fossil_heating_share",
            "renewable_share", 
            "no_energy_type",
        ]
        
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung", 
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Öfen",
            "keine_Heizung_share": "Keine Heizung",
        }
        
        energy_labels = {
            "fossil_heating_share": "Fossil",
            "renewable_share": "Erneuerbar",
            "no_energy_type": "Keine Angabe",
        }
        
        strict_result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels,
            threshold_dict=strict_thresholds
        )
        
        loose_result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels,
            threshold_dict=loose_thresholds
        )
        
        # Loose thresholds should generally include more areas
        assert len(loose_result) >= len(strict_result), \
            "Loose thresholds should include at least as many areas as strict thresholds"
    
    def test_pipeline_handles_missing_data(self, sample_geometries):
        """Test that pipeline handles missing or incomplete data gracefully."""
        # Create data with some missing values
        incomplete_renter_data = gpd.GeoDataFrame({
            'Eigentuemerquote': [40.0, None, 30.0, 45.0],  # One missing value
            'geometry': sample_geometries
        }, crs='EPSG:4326')
        
        incomplete_heating_data = gpd.GeoDataFrame({
            'Fernheizung': [10, 5, None, 12],  # One missing value
            'Etagenheizung': [20, 15, 12, 18],
            'Blockheizung': [5, 8, 6, 4],
            'Zentralheizung': [30, 25, 35, 28],
            'Einzel_Mehrraumoefen': [5, 7, 4, 6],
            'keine_Heizung': [0, 0, 0, 0],
            'geometry': sample_geometries
        }, crs='EPSG:4326')
        
        incomplete_energy_data = gpd.GeoDataFrame({
            'Gas': [40, 35, 30, 38],
            'Heizoel': [20, 15, 25, 18],
            'Holz_Holzpellets': [5, 8, 6, 7],
            'Biomasse_Biogas': [2, 3, 2, 2],
            'Solar_Geothermie_Waermepumpen': [8, 12, 10, 9],
            'Strom': [10, 15, 12, 11],
            'Kohle': [5, 2, 8, 3],
            'Fernwaerme': [8, 8, 5, 10],
            'kein_Energietraeger': [2, 2, 2, 2],
            'geometry': sample_geometries
        }, crs='EPSG:4326')
        
        threshold_dict = {
            "central_heating_thres": 0.4,
            "fossil_heating_thres": 0.4,
            "fernwaerme_thres": 0.1,
            "renter_share": 0.5
        }
        
        # Define the column lists and labels as in the prototype
        heating_typeshare_list = [
            "Fernheizung_share",
            "Etagenheizung_share", 
            "Blockheizung_share",
            "Zentralheizung_share",
            "Einzel_Mehrraumoefen_share",
            "keine_Heizung_share",
        ]
        
        energy_type_share_list = [
            "fossil_heating_share",
            "renewable_share", 
            "no_energy_type",
        ]
        
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung", 
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Öfen",
            "keine_Heizung_share": "Keine Heizung",
        }
        
        energy_labels = {
            "fossil_heating_share": "Fossil",
            "renewable_share": "Erneuerbar",
            "no_energy_type": "Keine Angabe",
        }
        
        # Should handle missing data without crashing
        result = get_rent_campaign_df(
            heating_type=incomplete_heating_data,
            energy_type=incomplete_energy_data,
            renter_df=incomplete_renter_data,
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels,
            threshold_dict=threshold_dict
        )
        
        # Should return valid result (possibly with fewer rows due to missing data)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) <= len(sample_geometries)


@pytest.mark.integration
@pytest.mark.medium
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_preprocessing_workflow_complete(self, temp_directory, sample_csv_content):
        """Test the complete preprocessing workflow with test data."""
        # Write test CSV file
        csv_file = temp_directory / "test_data.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(sample_csv_content)
        
        # Test full process_df workflow
        result_dict = process_df(
            path=str(temp_directory),
            sep=";",
            cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
            on_col="GITTER_ID_100m",
            drop_how="any",
            how="inner",
            gitter_id_column="GITTER_ID_100m"
        )
        
        assert len(result_dict) == 1
        result_gdf = result_dict['test_data']
        
        # Should be a GeoDataFrame
        assert isinstance(result_gdf, gpd.GeoDataFrame)
        
        # Should have converted values correctly (German decimal format)
        assert 'durchschnMieteQM' in result_gdf.columns
        assert result_gdf['durchschnMieteQM'].dtype == float
        
        # Should have proper geometry
        assert all(result_gdf.geometry.is_valid)
        assert result_gdf.crs == "EPSG:3035"
        
        # Should have dropped specified columns
        dropped_cols = ["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen", "GITTER_ID_100m"]
        for col in dropped_cols:
            assert col not in result_gdf.columns, f"Column {col} should have been dropped"
        
        # Should preserve data columns
        assert 'AnzahlWohnungen' in result_gdf.columns
    
    def test_preprocessing_with_special_values(self, temp_directory):
        """Test preprocessing with special values like KLAMMERN."""
        csv_content_with_special = """GITTER_ID_100m;x_mp_100m;y_mp_100m;durchschnMieteQM;AnzahlWohnungen;werterlaeuternde_Zeichen
CRS3035RES100mN2691700E4341100;4341150;2691750;6,26;3;
CRS3035RES100mN2692400E4341200;4341250;2692450;11,67;5;
CRS3035RES100mN2694800E4343900;4343950;2694850;4,45;3;KLAMMERN
CRS3035RES100mN2696700E4341400;4341450;2696750;6,58;5;"""
        
        csv_file = temp_directory / "test_special.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(csv_content_with_special)
        
        result_dict = process_df(
            path=str(temp_directory),
            sep=";",
            cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
            on_col="GITTER_ID_100m", 
            drop_how="any",
            how="inner",
            gitter_id_column="GITTER_ID_100m"
        )
        
        result_gdf = result_dict['test_special']
        
        # Should handle KLAMMERN rows correctly
        assert len(result_gdf) > 0
        assert all(result_gdf['durchschnMieteQM'] > 0)  # All should be valid numbers
    
    def test_load_geojson_folder_integration(self, temp_directory):
        """Test loading GeoJSON files from folder."""
        from src.functions import save_geodataframes
        
        # Create test GeoDataFrames
        test_gdf1 = gpd.GeoDataFrame({
            'value': [1, 2],
            'geometry': [
                gpd.points_from_xy([0, 1], [0, 1])[0],
                gpd.points_from_xy([0, 1], [0, 1])[1]
            ]
        }, crs='EPSG:4326')
        
        test_gdf2 = gpd.GeoDataFrame({
            'value': [3, 4], 
            'geometry': [
                gpd.points_from_xy([2, 3], [2, 3])[0],
                gpd.points_from_xy([2, 3], [2, 3])[1]
            ]
        }, crs='EPSG:4326')
        
        # Save as GeoJSON files
        save_geodataframes(
            {'dataset1': test_gdf1, 'dataset2': test_gdf2},
            output_dir=str(temp_directory),
            file_format='geojson'
        )
        
        # Load back using load_geojson_folder
        loaded_dict = load_geojson_folder(str(temp_directory))
        
        assert 'dataset1' in loaded_dict
        assert 'dataset2' in loaded_dict
        assert len(loaded_dict['dataset1']) == 2
        assert len(loaded_dict['dataset2']) == 2


@pytest.mark.integration  
@pytest.mark.medium
class TestWucherPipelineIntegration:
    """Integration tests for Wucher Miete detection pipeline."""
    
    def test_end_to_end_wucher_detection(self, temp_directory):
        """Test end-to-end Wucher detection from CSV to results."""
        from tests.utils.factories import CSVDataFactory, RentDataFactory
        
        # Create realistic test data with known outliers
        test_gdf = RentDataFactory.create_realistic_rent_sample(
            n_points=50,
            outlier_fraction=0.1  # 10% outliers
        )
        
        # Save as CSV format first (simulating preprocessing input)
        csv_content = CSVDataFactory.create_rent_csv_content(50)
        csv_file = temp_directory / "rent_test.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        # Run preprocessing
        processed_dict = process_df(
            path=str(temp_directory),
            sep=";",
            cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
            on_col="GITTER_ID_100m",
            drop_how="any", 
            how="inner",
            gitter_id_column="GITTER_ID_100m"
        )
        
        rent_gdf = processed_dict['rent_test']
        
        # Run Wucher detection
        wucher_results = detect_wucher_miete(
            rent_gdf,
            threshold=2.0,
            min_neighbors=5,
            min_rent_threshold=5.0
        )
        
        # Validate end-to-end results
        assert isinstance(wucher_results, gpd.GeoDataFrame)
        
        if len(wucher_results) > 0:
            assert 'wucher_miete_flag' in wucher_results.columns
            assert all(wucher_results['wucher_miete_flag'])
            assert 'durchschnMieteQM' in wucher_results.columns
            assert wucher_results.crs is not None
    
    def test_wucher_detection_with_real_parameters(self, sample_rent_gdf):
        """Test Wucher detection with real parameter values from params.py."""
        from params import WUCHER_DETECTION_PARAMS
        
        # Adjust parameters for small test dataset
        test_params = WUCHER_DETECTION_PARAMS.copy()
        test_params['min_neighbors'] = 2  # Lower for small test data
        test_params['threshold'] = 2.0     # Lower threshold for test
        
        result = detect_wucher_miete(sample_rent_gdf, **test_params)
        
        # Should work with real parameters (possibly returning empty result)
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'wucher_miete_flag' in result.columns or len(result) == 0
    
    def test_wucher_detection_output_format(self, outlier_test_grid):
        """Test that Wucher detection output has correct format for downstream use."""
        wucher_results = detect_wucher_miete(
            outlier_test_grid['gdf'],
            threshold=1.5,
            min_neighbors=2
        )
        
        if len(wucher_results) > 0:
            # Check output format compatibility
            required_columns = ['durchschnMieteQM', 'wucher_miete_flag', 'geometry']
            for col in required_columns:
                assert col in wucher_results.columns, f"Missing required column: {col}"
            
            # Check data types (can be float or int depending on test data)
            assert wucher_results['durchschnMieteQM'].dtype in [float, 'float64', int, 'int64']
            assert wucher_results['wucher_miete_flag'].dtype == bool
            
            # Check CRS preservation
            assert wucher_results.crs == outlier_test_grid['gdf'].crs
            
            # Check that all geometries are valid
            assert all(wucher_results.geometry.is_valid)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineScaling:
    """Test pipeline behavior with different data sizes."""
    
    def test_pipeline_scales_with_data_size(self, large_test_dataset):
        """Test that pipeline components scale appropriately with data size."""
        sizes = [50, 100, 200]
        processing_times = []
        
        for size in sizes:
            test_data = large_test_dataset(size)
            
            import time
            start_time = time.perf_counter()
            
            # Run a simplified pipeline
            result = detect_wucher_miete(
                test_data,
                threshold=2.0,
                min_neighbors=min(10, size // 10)  # Scale neighbors with data size
            )
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Should complete in reasonable time
            max_time = size / 10.0 + 5.0  # Rough scaling expectation
            assert processing_time < max_time, \
                f"Processing {size} points took too long: {processing_time:.2f}s"
        
        # Processing time should not increase exponentially
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Time should not increase more than 3x for 2x the data
            assert time_ratio <= size_ratio * 1.5, \
                f"Poor scaling: {time_ratio:.1f}x time for {size_ratio}x data"
