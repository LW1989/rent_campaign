#!/usr/bin/env python3
"""
Unit tests for core pipeline functions.

Migrated from unittest to pytest format with enhanced fixtures and parameterization.
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import (
    calc_total, get_heating_type, get_energy_type, get_renter_share,
    gitter_id_to_polygon, convert_to_float, drop_cols, create_geodataframe
)


@pytest.mark.unit
@pytest.mark.fast
class TestCoreFunctions:
    """Test core utility functions."""
    
    def test_calc_total_function(self):
        """Test the calc_total helper function."""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        result = calc_total(test_df, ['col1', 'col2'])
        
        assert 'total' in result.columns
        assert result['total'].tolist() == [5, 7, 9]
        assert len(result) == 3
    
    def test_calc_total_empty_columns(self):
        """Test calc_total with empty column list."""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = calc_total(test_df, [])
        
        assert 'total' in result.columns
        assert all(result['total'] == 0)
    
    def test_calc_total_single_column(self):
        """Test calc_total with single column."""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = calc_total(test_df, ['col1'])
        
        assert result['total'].tolist() == [1, 2, 3]


@pytest.mark.unit
@pytest.mark.fast  
class TestHeatingTypeProcessing:
    """Test heating type data processing."""
    
    def test_get_heating_type(self, sample_heating_type_data):
        """Test heating type processing with realistic data."""
        result = get_heating_type(sample_heating_type_data.copy())
        
        assert 'central_heating_share' in result.columns
        assert 'geometry' in result.columns
        assert len(result) == len(sample_heating_type_data)
        
        # Check that shares are calculated correctly (0-1 range)
        assert all(0 <= share <= 1 for share in result['central_heating_share'])
    
    def test_get_heating_type_preserves_geometry(self, sample_heating_type_data):
        """Test that geometry is preserved during processing."""
        original_geometry = sample_heating_type_data.geometry.copy()
        result = get_heating_type(sample_heating_type_data.copy())
        
        # Geometry should be preserved
        assert result.geometry.equals(original_geometry)
        assert result.crs == sample_heating_type_data.crs
    
    def test_get_heating_type_with_zeros(self, sample_geometries):
        """Test heating type processing with zero values."""
        zero_data = gpd.GeoDataFrame({
            'Fernheizung': [0, 0, 0, 0],
            'Etagenheizung': [0, 0, 0, 0], 
            'Blockheizung': [0, 0, 0, 0],
            'Zentralheizung': [0, 0, 0, 0],
            'Einzel_Mehrraumoefen': [0, 0, 0, 0],
            'keine_Heizung': [0, 0, 0, 0],
            'geometry': sample_geometries
        }, crs='EPSG:4326')
        
        result = get_heating_type(zero_data)
        
        # Should handle division by zero gracefully
        assert 'central_heating_share' in result.columns
        assert all(pd.isna(result['central_heating_share']) | 
                  (result['central_heating_share'] == 0))


@pytest.mark.unit
@pytest.mark.fast
class TestEnergyTypeProcessing:
    """Test energy type data processing."""
    
    def test_get_energy_type(self, sample_energy_type_data):
        """Test energy type processing with realistic data."""
        result = get_energy_type(sample_energy_type_data.copy())
        
        assert 'fossil_heating_share' in result.columns
        assert 'fernwaerme_share' in result.columns
        assert 'geometry' in result.columns
        assert len(result) == len(sample_energy_type_data)
        
        # Check that shares are valid percentages
        assert all(0 <= share <= 1 for share in result['fossil_heating_share'])
        assert all(0 <= share <= 1 for share in result['fernwaerme_share'])
    
    def test_get_energy_type_fossil_calculation(self, sample_energy_type_data):
        """Test that fossil fuel share is calculated correctly."""
        result = get_energy_type(sample_energy_type_data.copy())
        
        # Fossil fuels should include Gas + Heizoel + Kohle
        for i in range(len(result)):
            gas = sample_energy_type_data.iloc[i]['Gas']
            oil = sample_energy_type_data.iloc[i]['Heizoel'] 
            coal = sample_energy_type_data.iloc[i]['Kohle']
            # Sum only numeric columns (exclude geometry)
            numeric_cols = [col for col in sample_energy_type_data.columns if col != 'geometry']
            total = sample_energy_type_data.iloc[i][numeric_cols].sum()
            
            expected_fossil_share = (gas + oil + coal) / total
            actual_fossil_share = result.iloc[i]['fossil_heating_share']
            
            assert abs(expected_fossil_share - actual_fossil_share) < 0.01


@pytest.mark.unit
@pytest.mark.fast
class TestRenterShareProcessing:
    """Test renter share data processing."""
    
    def test_get_renter_share(self, sample_renter_data):
        """Test renter share processing with realistic data."""
        result = get_renter_share(sample_renter_data.copy())
        
        assert 'renter_share' in result.columns
        assert 'geometry' in result.columns
        assert len(result) == len(sample_renter_data)
        
        # Check conversion from ownership to renter percentage
        for i in range(len(result)):
            ownership = sample_renter_data.iloc[i]['Eigentuemerquote']
            expected_renter = (100 - ownership) / 100
            actual_renter = result.iloc[i]['renter_share']
            
            assert abs(expected_renter - actual_renter) < 0.01
    
    def test_get_renter_share_edge_cases(self, sample_geometries):
        """Test renter share with edge cases."""
        edge_case_data = gpd.GeoDataFrame({
            'Eigentuemerquote': [0.0, 100.0, 50.0, 25.5],  # Edge cases
            'geometry': sample_geometries
        }, crs='EPSG:4326')
        
        result = get_renter_share(edge_case_data)
        
        expected_renter_shares = [1.0, 0.0, 0.5, 0.745]  # (100 - ownership) / 100
        actual_renter_shares = result['renter_share'].tolist()
        
        for expected, actual in zip(expected_renter_shares, actual_renter_shares):
            assert abs(expected - actual) < 0.01


@pytest.mark.unit
@pytest.mark.fast 
class TestGitterIdProcessing:
    """Test GITTER_ID related functions."""
    
    def test_gitter_id_to_polygon_valid(self, real_gitter_ids):
        """Test GITTER_ID to polygon conversion with real IDs."""
        for gitter_id in real_gitter_ids:
            polygon = gitter_id_to_polygon(gitter_id)
            
            assert polygon is not None
            assert isinstance(polygon, Polygon)
            
            # Check that it's a 100m x 100m square
            bounds = polygon.bounds
            width = bounds[2] - bounds[0]  # max_x - min_x
            height = bounds[3] - bounds[1]  # max_y - min_y
            assert width == 100
            assert height == 100
    
    def test_gitter_id_to_polygon_invalid(self):
        """Test GITTER_ID to polygon with invalid formats."""
        invalid_ids = [
            "invalid_format",
            "CRS3035RES100m",  # Missing coordinates
            "CRS3035RES50mN2691700E4341100",  # Wrong resolution
            "",  # Empty string
            None  # None value
        ]
        
        for invalid_id in invalid_ids:
            if invalid_id is not None:  # Skip None test case
                result = gitter_id_to_polygon(invalid_id)
                assert result is None
    
    @pytest.mark.parametrize("gitter_id,expected_coords", [
        ("CRS3035RES100mN2691700E4341100", (4341100, 2691700)),
        ("CRS3035RES100mN2692400E4341200", (4341200, 2692400)),
    ])
    def test_gitter_id_coordinate_extraction(self, gitter_id, expected_coords):
        """Test that coordinates are extracted correctly from GITTER_ID."""
        polygon = gitter_id_to_polygon(gitter_id)
        
        assert polygon is not None
        
        # Check that the polygon's bottom-left corner matches expected coordinates
        bounds = polygon.bounds
        assert bounds[0] == expected_coords[0]  # min_x
        assert bounds[1] == expected_coords[1]  # min_y


@pytest.mark.unit
@pytest.mark.fast
class TestDataConversionFunctions:
    """Test data conversion and cleaning functions."""
    
    def test_convert_to_float_german_format(self):
        """Test German decimal format conversion."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': ['CRS3035RES100mN2691700E4341100', 'CRS3035RES100mN2691800E4341200'],
            'value1': ['1,5', '2,3'],  # German decimal format
            'value2': ['10,25', '20,75'],
            'text_col': ['text1', 'text2']
        })
        
        result = convert_to_float(test_df)
        
        # Check that decimal conversion worked
        assert result['value1'].iloc[0] == 1.5
        assert result['value1'].iloc[1] == 2.3
        assert result['value2'].iloc[0] == 10.25
        assert result['value2'].iloc[1] == 20.75
        
        # Check that GITTER_ID column remained unchanged
        assert result['GITTER_ID_100m'].iloc[0] == 'CRS3035RES100mN2691700E4341100'
        
        # Check that text columns were converted to numeric (should be NaN, then filled with 0)
        assert result['text_col'].iloc[0] == 0
    
    def test_convert_to_float_mixed_formats(self):
        """Test conversion with mixed formats."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': ['CRS3035RES100mN2691700E4341100'],
            'mixed_col': ['5,75'],  # German format
            'already_numeric': [3.14],  # Already numeric
            'integer_col': [42],  # Integer
            'missing_col': [np.nan]  # NaN value
        })
        
        result = convert_to_float(test_df)
        
        assert result['mixed_col'].iloc[0] == 5.75
        assert result['already_numeric'].iloc[0] == 3.14
        assert result['integer_col'].iloc[0] == 42
        assert result['missing_col'].iloc[0] == 0  # NaN should be filled with 0
    
    def test_drop_cols(self):
        """Test column dropping functionality."""
        test_df = pd.DataFrame({
            'keep1': [1, 2, 3],
            'keep2': [4, 5, 6], 
            'drop1': [7, 8, 9],
            'drop2': [10, 11, 12],
            'nonexistent': [13, 14, 15]
        })
        
        # Drop some columns including one that doesn't exist
        result = drop_cols(test_df, ['drop1', 'drop2', 'nonexistent_col'])
        
        # Check that correct columns remain
        expected_cols = ['keep1', 'keep2', 'nonexistent']
        assert list(result.columns) == expected_cols
        assert len(result) == 3
    
    def test_drop_cols_empty_list(self):
        """Test drop_cols with empty drop list."""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = drop_cols(test_df, [])
        
        # Should return original dataframe
        assert list(result.columns) == list(test_df.columns)
        assert result.equals(test_df)
    
    def test_create_geodataframe(self, real_gitter_ids):
        """Test creation of GeoDataFrame from real GITTER_IDs."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': real_gitter_ids + ['invalid_format'],  # Include invalid
            'value': list(range(len(real_gitter_ids) + 1))
        })
        
        result = create_geodataframe(test_df)
        
        # Should be a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        
        # Should have filtered out invalid GITTER_ID
        assert len(result) == len(real_gitter_ids)
        
        # Should have proper CRS
        assert result.crs == "EPSG:3035"
        
        # Should have geometry column
        assert 'geometry' in result.columns
        assert all(isinstance(geom, Polygon) for geom in result.geometry)
        
        # Should preserve other columns
        assert 'value' in result.columns
    
    def test_create_geodataframe_all_invalid(self):
        """Test create_geodataframe with all invalid GITTER_IDs."""
        test_df = pd.DataFrame({
            'GITTER_ID_100m': ['invalid1', 'invalid2', 'invalid3'],
            'value': [1, 2, 3]
        })
        
        result = create_geodataframe(test_df)
        
        # Should return empty GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0


class TestPieChartFunctionality:
    """Test the new heating_pie and energy_pie column functionality."""
    
    def test_make_pie_basic(self):
        """Test make_pie function with basic data."""
        from src.functions import make_pie
        
        # Create a sample row with share data
        sample_row = pd.Series({
            'Fernheizung_share': 0.2,
            'Etagenheizung_share': 0.15,
            'Blockheizung_share': 0.1,
            'Zentralheizung_share': 0.5,
            'Einzel_Mehrraumoefen_share': 0.05,
            'keine_Heizung_share': 0.0
        })
        
        cols = ['Fernheizung_share', 'Etagenheizung_share', 'Zentralheizung_share']
        labels = {
            'Fernheizung_share': 'Fernheizung',
            'Etagenheizung_share': 'Etagenheizung', 
            'Zentralheizung_share': 'Zentralheizung'
        }
        
        result = make_pie(sample_row, cols, labels)
        
        # Check that result is a list
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Check that each item is a dictionary with label and value
        for item in result:
            assert isinstance(item, dict)
            assert 'label' in item
            assert 'value' in item
            assert isinstance(item['value'], (int, float))
        
        # Check specific values
        assert result[0]['label'] == 'Fernheizung'
        assert result[0]['value'] == 0.2
        assert result[1]['label'] == 'Etagenheizung'
        assert result[1]['value'] == 0.15
        assert result[2]['label'] == 'Zentralheizung'
        assert result[2]['value'] == 0.5
    
    def test_get_rent_campaign_df_includes_pie_columns(self, sample_heating_type_data, sample_energy_type_data, sample_renter_data, sample_threshold_dict):
        """Test that get_rent_campaign_df includes the new heating_pie and energy_pie columns."""
        from src.functions import get_rent_campaign_df
        
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
        
        # Check that pie columns exist
        assert 'heating_pie' in result.columns
        assert 'energy_pie' in result.columns
        
        # Check that pie columns contain lists
        assert all(isinstance(x, list) for x in result['heating_pie'])
        assert all(isinstance(x, list) for x in result['energy_pie'])
        
        # Check that each pie entry is a list of dictionaries with label and value
        for pie_list in result['heating_pie']:
            assert all(isinstance(item, dict) for item in pie_list)
            for item in pie_list:
                assert 'label' in item
                assert 'value' in item
                assert isinstance(item['value'], (int, float))
                assert 0 <= item['value'] <= 1
        
        for pie_list in result['energy_pie']:
            assert all(isinstance(item, dict) for item in pie_list)
            for item in pie_list:
                assert 'label' in item
                assert 'value' in item
                assert isinstance(item['value'], (int, float))
                assert 0 <= item['value'] <= 1
    
    def test_pie_chart_values_consistency(self, sample_heating_type_data, sample_energy_type_data, sample_renter_data, sample_threshold_dict):
        """Test that pie chart values are consistent with input data."""
        from src.functions import get_rent_campaign_df
        
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
        
        # For each row, check that the pie chart values make sense
        for idx, row in result.iterrows():
            heating_pie = row['heating_pie']
            energy_pie = row['energy_pie']
            
            # All values should be non-negative
            for item in heating_pie + energy_pie:
                assert item['value'] >= 0
                assert item['value'] <= 1
