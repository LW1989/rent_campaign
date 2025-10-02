#!/usr/bin/env python3
"""
Unit tests for etagenheizung_flag feature.

Tests the new etagenheizung_flag functionality including:
- Flag calculation based on threshold
- Integration with get_rent_campaign_df()
- Correct handling of edge cases
- THRESHOLD_PARAMS configuration
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
    calc_rent_campaign_flags,
    get_rent_campaign_df,
    get_heating_type
)
from params import THRESHOLD_PARAMS


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungFlagCalculation:
    """Test etagenheizung_flag calculation in calc_rent_campaign_flags()."""
    
    def test_etagenheizung_flag_above_threshold(self, sample_geometries):
        """Test that etagenheizung_flag is True when Etagenheizung_share > threshold."""
        # Create test data with Etagenheizung_share = 0.7 (above 0.6 threshold)
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],  # High enough to pass renter filter
            'Etagenheizung_share': [0.7, 0.75],  # Above threshold
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert 'etagenheizung_flag' in result.columns
        assert all(result['etagenheizung_flag'] == True)
    
    def test_etagenheizung_flag_below_threshold(self, sample_geometries):
        """Test that etagenheizung_flag is False when Etagenheizung_share < threshold."""
        # Create test data with Etagenheizung_share = 0.4 (below 0.6 threshold)
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],  # High enough to pass renter filter
            'Etagenheizung_share': [0.4, 0.3],  # Below threshold
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert 'etagenheizung_flag' in result.columns
        assert all(result['etagenheizung_flag'] == False)
    
    def test_etagenheizung_flag_at_threshold(self, sample_geometries):
        """Test that etagenheizung_flag is False when Etagenheizung_share == threshold."""
        # Create test data with Etagenheizung_share = 0.6 (exactly at threshold)
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [0.6, 0.6],  # Exactly at threshold
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        # Should be False because we use '>' not '>='
        assert all(result['etagenheizung_flag'] == False)
    
    def test_etagenheizung_flag_mixed_values(self, sample_geometries):
        """Test etagenheizung_flag with mixed values above and below threshold."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5, 0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5, 0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1, 0.1, 0.1],
            'renter_share': [0.7, 0.7, 0.7, 0.7],
            'Etagenheizung_share': [0.7, 0.4, 0.8, 0.5],  # Mixed values
            'geometry': sample_geometries
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        expected_flags = [True, False, True, False]
        assert result['etagenheizung_flag'].tolist() == expected_flags


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungFlagWithNaNValues:
    """Test etagenheizung_flag handling of NaN and edge cases."""
    
    def test_etagenheizung_flag_with_nan(self, sample_geometries):
        """Test that etagenheizung_flag handles NaN values correctly."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [np.nan, 0.7],  # One NaN value
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        # NaN > 0.6 should be False in pandas
        assert result['etagenheizung_flag'].iloc[0] == False
        assert result['etagenheizung_flag'].iloc[1] == True
    
    def test_etagenheizung_flag_with_zero(self, sample_geometries):
        """Test that etagenheizung_flag handles zero values correctly."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [0.0, 0.0],  # Zero values
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert all(result['etagenheizung_flag'] == False)
    
    def test_etagenheizung_flag_with_one_value(self, sample_geometries):
        """Test that etagenheizung_flag handles 1.0 (100%) correctly."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [1.0, 1.0],  # 100% Etagenheizung
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert all(result['etagenheizung_flag'] == True)


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungFlagInPipeline:
    """Test etagenheizung_flag integration with get_rent_campaign_df()."""
    
    def test_etagenheizung_flag_in_output_columns(self, sample_heating_type_data, 
                                                    sample_energy_type_data, 
                                                    sample_renter_data):
        """Test that etagenheizung_flag appears in get_rent_campaign_df output."""
        heating_typeshare_list = [
            "Fernheizung_share", "Etagenheizung_share", "Blockheizung_share",
            "Zentralheizung_share", "Einzel_Mehrraumoefen_share"
        ]
        energy_type_share_list = [
            "fossil_heating_share", "renewable_share", "fernwaerme_share"
        ]
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung",
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Einzel-/Mehrraumöfen"
        }
        energy_labels = {
            "fossil_heating_share": "Fossile Brennstoffe",
            "renewable_share": "Erneuerbare Energien",
            "fernwaerme_share": "Fernwärme"
        }
        
        # Reproject to same CRS
        sample_renter_data = sample_renter_data.to_crs('EPSG:3035')
        sample_heating_type_data = sample_heating_type_data.to_crs('EPSG:3035')
        sample_energy_type_data = sample_energy_type_data.to_crs('EPSG:3035')
        
        result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels
        )
        
        # Check that etagenheizung_flag is in the output
        assert 'etagenheizung_flag' in result.columns
    
    def test_etagenheizung_flag_position_in_columns(self, sample_heating_type_data,
                                                      sample_energy_type_data,
                                                      sample_renter_data):
        """Test that etagenheizung_flag is in the correct position after renter_flag."""
        heating_typeshare_list = [
            "Fernheizung_share", "Etagenheizung_share", "Blockheizung_share",
            "Zentralheizung_share", "Einzel_Mehrraumoefen_share"
        ]
        energy_type_share_list = [
            "fossil_heating_share", "renewable_share", "fernwaerme_share"
        ]
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung",
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Einzel-/Mehrraumöfen"
        }
        energy_labels = {
            "fossil_heating_share": "Fossile Brennstoffe",
            "renewable_share": "Erneuerbare Energien",
            "fernwaerme_share": "Fernwärme"
        }
        
        # Reproject to same CRS
        sample_renter_data = sample_renter_data.to_crs('EPSG:3035')
        sample_heating_type_data = sample_heating_type_data.to_crs('EPSG:3035')
        sample_energy_type_data = sample_energy_type_data.to_crs('EPSG:3035')
        
        result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels
        )
        
        columns = result.columns.tolist()
        
        # Check that etagenheizung_flag comes after renter_flag
        assert 'renter_flag' in columns
        assert 'etagenheizung_flag' in columns
        
        renter_flag_idx = columns.index('renter_flag')
        etagenheizung_flag_idx = columns.index('etagenheizung_flag')
        
        assert etagenheizung_flag_idx == renter_flag_idx + 1
    
    def test_etagenheizung_share_not_in_output(self, sample_heating_type_data,
                                                 sample_energy_type_data,
                                                 sample_renter_data):
        """Test that Etagenheizung_share is dropped from final output."""
        heating_typeshare_list = [
            "Fernheizung_share", "Etagenheizung_share", "Blockheizung_share",
            "Zentralheizung_share", "Einzel_Mehrraumoefen_share"
        ]
        energy_type_share_list = [
            "fossil_heating_share", "renewable_share", "fernwaerme_share"
        ]
        heating_labels = {
            "Fernheizung_share": "Fernheizung",
            "Etagenheizung_share": "Etagenheizung",
            "Blockheizung_share": "Blockheizung",
            "Zentralheizung_share": "Zentralheizung",
            "Einzel_Mehrraumoefen_share": "Einzel-/Mehrraumöfen"
        }
        energy_labels = {
            "fossil_heating_share": "Fossile Brennstoffe",
            "renewable_share": "Erneuerbare Energien",
            "fernwaerme_share": "Fernwärme"
        }
        
        # Reproject to same CRS
        sample_renter_data = sample_renter_data.to_crs('EPSG:3035')
        sample_heating_type_data = sample_heating_type_data.to_crs('EPSG:3035')
        sample_energy_type_data = sample_energy_type_data.to_crs('EPSG:3035')
        
        result = get_rent_campaign_df(
            heating_type=sample_heating_type_data.copy(),
            energy_type=sample_energy_type_data.copy(),
            renter_df=sample_renter_data.copy(),
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels
        )
        
        # Etagenheizung_share should be dropped (only flag remains)
        assert 'Etagenheizung_share' not in result.columns
        assert 'etagenheizung_flag' in result.columns


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungFlagThresholdConfiguration:
    """Test etagenheizung_flag uses THRESHOLD_PARAMS configuration."""
    
    def test_threshold_params_contains_etagenheizung(self):
        """Test that THRESHOLD_PARAMS includes etagenheizung_thres."""
        assert 'etagenheizung_thres' in THRESHOLD_PARAMS
    
    def test_threshold_params_etagenheizung_value(self):
        """Test that etagenheizung_thres has expected value."""
        assert THRESHOLD_PARAMS['etagenheizung_thres'] == 0.6
    
    def test_custom_threshold_value(self, sample_geometries):
        """Test that etagenheizung_flag respects custom threshold value."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [0.5, 0.5],  # 50%
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        # Use custom threshold of 0.4 (40%)
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.4  # Custom lower threshold
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        # With 0.4 threshold, 0.5 should be True
        assert all(result['etagenheizung_flag'] == True)
        
        # With 0.6 threshold, 0.5 should be False
        threshold_dict['etagenheizung_thres'] = 0.6
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        assert all(result['etagenheizung_flag'] == False)


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungShareCalculation:
    """Test that Etagenheizung_share is calculated correctly from raw data."""
    
    def test_heating_type_creates_etagenheizung_share(self, sample_heating_type_data):
        """Test that get_heating_type creates Etagenheizung_share column."""
        result = get_heating_type(sample_heating_type_data.copy())
        
        assert 'Etagenheizung_share' in result.columns
    
    def test_etagenheizung_share_calculation(self, sample_geometries):
        """Test that Etagenheizung_share is calculated correctly."""
        # Create data with known values
        heating_data = gpd.GeoDataFrame({
            'Fernheizung': [10],
            'Etagenheizung': [60],  # 60 out of 100
            'Blockheizung': [10],
            'Zentralheizung': [20],
            'Einzel_Mehrraumoefen': [0],
            'keine_Heizung': [0],
            'geometry': [sample_geometries[0]]
        }, crs='EPSG:3035')
        
        result = get_heating_type(heating_data)
        
        # Should be 60 / 100 = 0.6
        assert result['Etagenheizung_share'].iloc[0] == 0.6
    
    def test_etagenheizung_share_range(self, sample_heating_type_data):
        """Test that Etagenheizung_share is in valid range [0, 1]."""
        result = get_heating_type(sample_heating_type_data.copy())
        
        assert all(0 <= share <= 1 for share in result['Etagenheizung_share'])


@pytest.mark.unit
@pytest.mark.fast
class TestEtagenheizungFlagDataTypes:
    """Test data types and schema of etagenheizung_flag."""
    
    def test_etagenheizung_flag_is_boolean(self, sample_geometries):
        """Test that etagenheizung_flag has boolean dtype."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [0.7, 0.4],
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert result['etagenheizung_flag'].dtype == bool
    
    def test_etagenheizung_flag_no_missing_values(self, sample_geometries):
        """Test that etagenheizung_flag has no NaN values after calculation."""
        test_df = gpd.GeoDataFrame({
            'central_heating_share': [0.5, 0.5],
            'fossil_heating_share': [0.5, 0.5],
            'fernwaerme_share': [0.1, 0.1],
            'renter_share': [0.7, 0.7],
            'Etagenheizung_share': [0.7, 0.4],
            'geometry': sample_geometries[:2]
        }, crs='EPSG:3035')
        
        threshold_dict = {
            "central_heating_thres": 0.6,
            "fossil_heating_thres": 0.6,
            "fernwaerme_thres": 0.2,
            "renter_share": 0.6,
            "etagenheizung_thres": 0.6
        }
        
        result = calc_rent_campaign_flags(test_df.copy(), threshold_dict)
        
        assert result['etagenheizung_flag'].notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

