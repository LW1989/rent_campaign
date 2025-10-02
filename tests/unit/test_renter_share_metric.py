"""
Unit tests for renter_share metric in metric cards.

Tests the addition of renter_share to the rent campaign analysis output
and its inclusion in the metric card system.
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

from src.functions import (
    get_rent_campaign_df, 
    get_renter_share,
    create_metric_card,
    add_metric_cards_to_districts,
    calculate_city_means
)
from params import METRIC_CARD_CONFIG


@pytest.mark.unit
@pytest.mark.fast
class TestRenterShareInRentCampaignDf:
    """Test that renter_share is included in rent_campaign_df output."""
    
    def test_renter_share_in_output_columns(self):
        """Test that renter_share column is present in get_rent_campaign_df output."""
        # Create test data with minimal required columns
        heating_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Fernheizung': [10, 20],
            'Etagenheizung': [5, 10],
            'Blockheizung': [3, 5],
            'Zentralheizung': [30, 40],
            'Einzel_Mehrraumoefen': [2, 5],
            'keine_Heizung': [0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        energy_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Gas': [20, 30],
            'Heizoel': [10, 15],
            'Kohle': [5, 5],
            'Fernwaerme': [10, 20],
            'Strom': [5, 10],
            'Holz_Holzpellets': [2, 3],
            'Biomasse_Biogas': [1, 1],
            'Solar_Geothermie_Waermepumpen': [2, 3],
            'kein_Energietraeger': [0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        renter_df = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Eigentuemerquote': [30, 40],  # 30% owners = 70% renters
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        # Define share lists and labels
        heating_typeshare_list = ['Fernheizung_share', 'Etagenheizung_share', 
                                  'Blockheizung_share', 'Zentralheizung_share', 
                                  'Einzel_Mehrraumoefen_share']
        energy_type_share_list = ['fossil_heating_share', 'renewable_share', 
                                  'fernwaerme_share']
        heating_labels = {
            'Fernheizung_share': 'Fernheizung',
            'Etagenheizung_share': 'Etagenheizung',
            'Blockheizung_share': 'Blockheizung',
            'Zentralheizung_share': 'Zentralheizung',
            'Einzel_Mehrraumoefen_share': 'Einzel-/Mehrraumöfen'
        }
        energy_labels = {
            'fossil_heating_share': 'Fossile Brennstoffe',
            'renewable_share': 'Erneuerbare Energien',
            'fernwaerme_share': 'Fernwärme'
        }
        
        # Call get_rent_campaign_df
        result = get_rent_campaign_df(
            heating_type=heating_type,
            energy_type=energy_type,
            renter_df=renter_df,
            heating_typeshare_list=heating_typeshare_list,
            energy_type_share_list=energy_type_share_list,
            heating_labels=heating_labels,
            energy_labels=energy_labels,
            threshold_dict={
                'central_heating_thres': 0.6,
                'fossil_heating_thres': 0.6,
                'fernwaerme_thres': 0.2,
                'renter_share': 0.6,
                'etagenheizung_thres': 0.6
            }
        )
        
        # Verify renter_share column exists
        assert 'renter_share' in result.columns, \
            f"renter_share not in columns: {list(result.columns)}"
    
    def test_renter_share_correct_values(self):
        """Test that renter_share values are correctly calculated."""
        # Create test data
        heating_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2', 'sq3'],
            'Fernheizung': [10, 20, 30],
            'Etagenheizung': [5, 10, 15],
            'Blockheizung': [3, 5, 7],
            'Zentralheizung': [30, 40, 50],
            'Einzel_Mehrraumoefen': [2, 5, 8],
            'keine_Heizung': [0, 0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
            ]
        }, crs='EPSG:3035')
        
        energy_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2', 'sq3'],
            'Gas': [20, 30, 40],
            'Heizoel': [10, 15, 20],
            'Kohle': [5, 5, 5],
            'Fernwaerme': [10, 20, 30],
            'Strom': [5, 10, 15],
            'Holz_Holzpellets': [2, 3, 4],
            'Biomasse_Biogas': [1, 1, 1],
            'Solar_Geothermie_Waermepumpen': [2, 3, 4],
            'kein_Energietraeger': [0, 0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
            ]
        }, crs='EPSG:3035')
        
        renter_df = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2', 'sq3'],
            'Eigentuemerquote': [30, 40, 50],  # 70%, 60%, 50% renters
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
            ]
        }, crs='EPSG:3035')
        
        # Call get_rent_campaign_df
        result = get_rent_campaign_df(
            heating_type=heating_type,
            energy_type=energy_type,
            renter_df=renter_df,
            heating_typeshare_list=['Fernheizung_share', 'Etagenheizung_share', 
                                   'Blockheizung_share', 'Zentralheizung_share', 
                                   'Einzel_Mehrraumoefen_share'],
            energy_type_share_list=['fossil_heating_share', 'renewable_share', 
                                   'fernwaerme_share'],
            heating_labels={
                'Fernheizung_share': 'Fernheizung',
                'Etagenheizung_share': 'Etagenheizung',
                'Blockheizung_share': 'Blockheizung',
                'Zentralheizung_share': 'Zentralheizung',
                'Einzel_Mehrraumoefen_share': 'Einzel-/Mehrraumöfen'
            },
            energy_labels={
                'fossil_heating_share': 'Fossile Brennstoffe',
                'renewable_share': 'Erneuerbare Energien',
                'fernwaerme_share': 'Fernwärme'
            },
            threshold_dict={
                'central_heating_thres': 0.6,
                'fossil_heating_thres': 0.6,
                'fernwaerme_thres': 0.2,
                'renter_share': 0.6,
                'etagenheizung_thres': 0.6
            }
        )
        
        # Verify renter_share values (70%, 60%, 50% as decimals)
        expected_values = [0.70, 0.60, 0.50]
        actual_values = result['renter_share'].values
        
        for i, (expected, actual) in enumerate(zip(expected_values, actual_values)):
            assert abs(expected - actual) < 0.01, \
                f"Square {i}: Expected {expected}, got {actual}"
    
    def test_renter_share_with_renter_flag(self):
        """Test that renter_share is consistent with renter_flag."""
        # Create test data with values above threshold (only those are returned)
        heating_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Fernheizung': [10, 20],
            'Etagenheizung': [5, 10],
            'Blockheizung': [3, 5],
            'Zentralheizung': [30, 40],
            'Einzel_Mehrraumoefen': [2, 5],
            'keine_Heizung': [0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        energy_type = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Gas': [20, 30],
            'Heizoel': [10, 15],
            'Kohle': [5, 5],
            'Fernwaerme': [10, 20],
            'Strom': [5, 10],
            'Holz_Holzpellets': [2, 3],
            'Biomasse_Biogas': [1, 1],
            'Solar_Geothermie_Waermepumpen': [2, 3],
            'kein_Energietraeger': [0, 0],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        renter_df = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'Eigentuemerquote': [30, 35],  # 70% and 65% renters (both above 60%)
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        # Call with threshold of 0.6 (60%)
        result = get_rent_campaign_df(
            heating_type=heating_type,
            energy_type=energy_type,
            renter_df=renter_df,
            heating_typeshare_list=['Fernheizung_share', 'Etagenheizung_share',
                                   'Blockheizung_share', 'Zentralheizung_share',
                                   'Einzel_Mehrraumoefen_share'],
            energy_type_share_list=['fossil_heating_share', 'renewable_share',
                                   'fernwaerme_share'],
            heating_labels={
                'Fernheizung_share': 'Fernheizung',
                'Etagenheizung_share': 'Etagenheizung',
                'Blockheizung_share': 'Blockheizung',
                'Zentralheizung_share': 'Zentralheizung',
                'Einzel_Mehrraumoefen_share': 'Einzel-/Mehrraumöfen'
            },
            energy_labels={
                'fossil_heating_share': 'Fossile Brennstoffe',
                'renewable_share': 'Erneuerbare Energien',
                'fernwaerme_share': 'Fernwärme'
            },
            threshold_dict={
                'central_heating_thres': 0.6,
                'fossil_heating_thres': 0.6,
                'fernwaerme_thres': 0.2,
                'renter_share': 0.6,
                'etagenheizung_thres': 0.6
            }
        )
        
        # Verify consistency between renter_share and renter_flag
        # Both squares should pass threshold and have renter_flag=True
        # (Note: calc_rent_campaign_flags filters to only return renter_flag==True)
        assert len(result) == 2, "Both squares should pass the renter threshold"
        assert result['renter_share'].iloc[0] == 0.70
        assert result['renter_flag'].iloc[0] == True, \
            "Square 1 with 70% renters should have renter_flag=True"
        
        assert result['renter_share'].iloc[1] == 0.65
        assert result['renter_flag'].iloc[1] == True, \
            "Square 2 with 65% renters should have renter_flag=True"


@pytest.mark.unit
@pytest.mark.fast
class TestRenterShareInMetricCardConfig:
    """Test that renter_share is properly configured in METRIC_CARD_CONFIG."""
    
    def test_renter_share_in_config(self):
        """Test that METRIC_CARD_CONFIG includes renter_share configuration."""
        # Verify renter_share is in the config
        assert 'renter_share' in METRIC_CARD_CONFIG, \
            "renter_share not found in METRIC_CARD_CONFIG"
    
    def test_renter_share_config_structure(self):
        """Test that renter_share config has correct structure."""
        renter_config = METRIC_CARD_CONFIG['renter_share']
        
        # Verify required keys
        assert 'id' in renter_config, "Missing 'id' key in renter_share config"
        assert 'label' in renter_config, "Missing 'label' key in renter_share config"
    
    def test_renter_share_config_values(self):
        """Test that renter_share config has correct values."""
        renter_config = METRIC_CARD_CONFIG['renter_share']
        
        # Verify id
        assert renter_config['id'] == 'renter_share', \
            f"Expected id='renter_share', got '{renter_config['id']}'"
        
        # Verify label (German for "Renter Share")
        assert renter_config['label'] == 'Mieterquote', \
            f"Expected label='Mieterquote', got '{renter_config['label']}'"


@pytest.mark.unit
@pytest.mark.fast
class TestRenterShareMetricCard:
    """Test metric card creation for renter_share."""
    
    def test_create_metric_card_for_renter_share_above_average(self):
        """Test metric card creation when square is above city average."""
        # Square has 70% renters, city average is 60%
        metric_card = create_metric_card(
            value=0.70,
            group_mean=0.60,
            metric_id='renter_share',
            metric_label='Mieterquote'
        )
        
        # Verify structure
        assert metric_card['id'] == 'renter_share'
        assert metric_card['label'] == 'Mieterquote'
        assert metric_card['value'] == 0.70
        assert metric_card['group_mean'] == 0.60
        
        # Verify calculated fields
        assert abs(metric_card['abs_diff'] - 0.10) < 0.001  # 0.70 - 0.60
        assert abs(metric_card['pct_diff'] - 0.1667) < 0.01  # (0.70/0.60) - 1
        assert metric_card['direction'] == 'above'
    
    def test_create_metric_card_for_renter_share_below_average(self):
        """Test metric card creation when square is below city average."""
        # Square has 50% renters, city average is 60%
        metric_card = create_metric_card(
            value=0.50,
            group_mean=0.60,
            metric_id='renter_share',
            metric_label='Mieterquote'
        )
        
        # Verify calculated fields
        assert abs(metric_card['abs_diff'] - (-0.10)) < 0.001  # 0.50 - 0.60
        assert abs(metric_card['pct_diff'] - (-0.1667)) < 0.01  # (0.50/0.60) - 1
        assert metric_card['direction'] == 'below'
    
    def test_create_metric_card_for_renter_share_equal_average(self):
        """Test metric card creation when square equals city average."""
        # Square has 60% renters, city average is 60%
        metric_card = create_metric_card(
            value=0.60,
            group_mean=0.60,
            metric_id='renter_share',
            metric_label='Mieterquote'
        )
        
        # Verify calculated fields
        assert abs(metric_card['abs_diff']) < 0.001  # 0.60 - 0.60 = 0
        assert abs(metric_card['pct_diff']) < 0.001  # (0.60/0.60) - 1 = 0
        assert metric_card['direction'] == 'equal'
    
    def test_create_metric_card_with_nan_value(self):
        """Test metric card creation with NaN renter_share value."""
        metric_card = create_metric_card(
            value=np.nan,
            group_mean=0.60,
            metric_id='renter_share',
            metric_label='Mieterquote'
        )
        
        # Should handle NaN gracefully
        assert metric_card['id'] == 'renter_share'
        assert pd.isna(metric_card['value'])
        assert metric_card['abs_diff'] is None
        assert metric_card['pct_diff'] is None
        assert metric_card['direction'] == 'equal'


@pytest.mark.unit
@pytest.mark.fast
class TestRenterShareInMetricCardsPipeline:
    """Integration test for renter_share in the full metric cards pipeline."""
    
    def test_metric_cards_include_renter_share(self):
        """Test that renter_share metric cards are created in the full pipeline."""
        # Create mock rent_campaign_df with renter_share
        rent_campaign_df = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2', 'sq3'],
            'renter_share': [0.70, 0.65, 0.55],  # As decimals
            'durchschnMieteQM': [12.0, 11.0, 10.0],
            'AnteilUeber65': [0.15, 0.20, 0.25],
            'AnteilAuslaender': [0.10, 0.15, 0.20],
            'durchschnFlaechejeBew': [35.0, 40.0, 45.0],
            'Einwohner': [100, 80, 90],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
            ]
        }, crs='EPSG:3035')
        
        # Create mock city boundaries (covering all squares)
        krs_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(-1, -1), (4, -1), (4, 2), (-1, 2)])]
        }, crs='EPSG:3035')
        
        # Create mock results_dict
        results_dict = {
            'test_district': rent_campaign_df.iloc[[0, 1]].copy()
        }
        
        # Create mock district_city_mapping
        district_city_mapping = {
            'test_district': {'city_name': 'Test City'}
        }
        
        # Calculate city means
        metric_columns = list(METRIC_CARD_CONFIG.keys())
        city_means = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        # Add metric cards
        enhanced_results = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, METRIC_CARD_CONFIG
        )
        
        # Verify metric cards were added
        enhanced_gdf = enhanced_results['test_district']
        assert 'metric_cards' in enhanced_gdf.columns, \
            "metric_cards column not found in enhanced results"
        
        # Get first square's metric cards
        first_cards = enhanced_gdf['metric_cards'].iloc[0]
        
        # Verify renter_share metric card exists
        assert 'renter_share' in first_cards, \
            f"renter_share not in metric cards. Available: {list(first_cards.keys())}"
        
        # Verify renter_share metric card content
        renter_card = first_cards['renter_share']
        assert renter_card['id'] == 'renter_share'
        assert renter_card['label'] == 'Mieterquote'
        assert renter_card['value'] == 0.70  # First square has 70% renters
        assert 'group_mean' in renter_card
        assert 'abs_diff' in renter_card
        assert 'pct_diff' in renter_card
        assert 'direction' in renter_card
    
    def test_renter_share_metric_values_in_pipeline(self):
        """Test that renter_share metric card has correct comparison values."""
        # Create test data with known values
        rent_campaign_df = gpd.GeoDataFrame({
            'GITTER_ID_100m': ['sq1', 'sq2'],
            'renter_share': [0.80, 0.50],  # 80% and 50% renters
            'durchschnMieteQM': [10.0, 10.0],
            'AnteilUeber65': [0.15, 0.15],
            'AnteilAuslaender': [0.10, 0.10],
            'durchschnFlaechejeBew': [35.0, 35.0],
            'Einwohner': [100, 100],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
            ]
        }, crs='EPSG:3035')
        
        krs_gdf = gpd.GeoDataFrame({
            'GEN': ['Test City'],
            'BEZ': ['Kreisfreie Stadt'],
            'geometry': [Polygon([(-1, -1), (3, -1), (3, 2), (-1, 2)])]
        }, crs='EPSG:3035')
        
        results_dict = {
            'test_district': rent_campaign_df.copy()
        }
        
        district_city_mapping = {
            'test_district': {'city_name': 'Test City'}
        }
        
        # Calculate city means
        metric_columns = list(METRIC_CARD_CONFIG.keys())
        city_means = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        # City mean should be (0.80 + 0.50) / 2 = 0.65
        expected_city_mean = 0.65
        assert abs(city_means['Test City']['renter_share'] - expected_city_mean) < 0.01
        
        # Add metric cards
        enhanced_results = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, METRIC_CARD_CONFIG
        )
        
        enhanced_gdf = enhanced_results['test_district']
        
        # Check first square (0.80 > 0.65 → above average)
        first_cards = enhanced_gdf['metric_cards'].iloc[0]
        first_renter_card = first_cards['renter_share']
        assert first_renter_card['value'] == 0.80
        assert abs(first_renter_card['group_mean'] - 0.65) < 0.01
        assert first_renter_card['direction'] == 'above'
        
        # Check second square (0.50 < 0.65 → below average)
        second_cards = enhanced_gdf['metric_cards'].iloc[1]
        second_renter_card = second_cards['renter_share']
        assert second_renter_card['value'] == 0.50
        assert abs(second_renter_card['group_mean'] - 0.65) < 0.01
        assert second_renter_card['direction'] == 'below'

