"""
Integration tests for Wucher Miete join fix.

These tests verify that the attribute-based merge on GITTER_ID_100m 
correctly fixes the spatial join bug that was causing data misalignment.
"""

import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box

# Import the function we're testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline import integrate_wucher_detection
from src.functions import detect_wucher_miete
from params import WUCHER_DETECTION_PARAMS


@pytest.fixture
def grid_squares_with_ids():
    """Create a grid of squares with GITTER_ID_100m for testing."""
    squares = []
    gitter_ids = []
    
    # Create 5x5 grid of 100m squares
    for i in range(5):
        for j in range(5):
            x = 4000000 + j * 100
            y = 3000000 + i * 100
            geom = box(x, y, x + 100, y + 100)
            gitter_id = f"CRS3035RES100mN{y}E{x}"
            squares.append(geom)
            gitter_ids.append(gitter_id)
    
    gdf = gpd.GeoDataFrame({
        'GITTER_ID_100m': gitter_ids,
        'geometry': squares
    }, crs='EPSG:3035')
    
    return gdf


@pytest.fixture
def rent_data_with_ids(grid_squares_with_ids):
    """Create rent data matching the grid squares."""
    rent_gdf = grid_squares_with_ids.copy()
    
    # Add varying rent values
    np.random.seed(42)
    rent_values = np.random.uniform(5.0, 15.0, len(rent_gdf))
    rent_gdf['durchschnMieteQM'] = rent_values
    rent_gdf['AnzahlWohnungen'] = np.random.randint(5, 50, len(rent_gdf))
    
    return rent_gdf


@pytest.fixture
def campaign_data_with_ids(grid_squares_with_ids):
    """Create campaign data with flags."""
    campaign_gdf = grid_squares_with_ids.copy()
    
    # Add campaign flags
    campaign_gdf['central_heating_flag'] = True
    campaign_gdf['fossil_heating_flag'] = True
    campaign_gdf['fernwaerme_flag'] = False
    campaign_gdf['renter_flag'] = True
    
    return campaign_gdf


@pytest.mark.integration
@pytest.mark.fast
class TestWucherJoinFix:
    """Test suite for verifying the Wucher join fix."""
    
    def test_no_duplicates_created(self, campaign_data_with_ids, rent_data_with_ids):
        """Verify that attribute-based merge does not create duplicates."""
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data_with_ids
        }
        
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # Should have same number of rows as input
        assert len(result) == len(campaign_data_with_ids), \
            f"Expected {len(campaign_data_with_ids)} rows, got {len(result)}"
        
        # Should have no duplicate GITTER_IDs
        assert result['GITTER_ID_100m'].is_unique, "Found duplicate GITTER_IDs in result"
    
    def test_rent_values_match_exactly(self, campaign_data_with_ids, rent_data_with_ids):
        """Verify that rent values match the source data exactly."""
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data_with_ids
        }
        
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # Create a mapping of expected rent values
        expected_rents = dict(zip(
            rent_data_with_ids['GITTER_ID_100m'], 
            rent_data_with_ids['durchschnMieteQM']
        ))
        
        # Check each result row has correct rent
        for idx, row in result.iterrows():
            gitter_id = row['GITTER_ID_100m']
            result_rent = row['durchschnMieteQM']
            expected_rent = expected_rents.get(gitter_id)
            
            if pd.notna(result_rent) and expected_rent is not None:
                assert result_rent == expected_rent, \
                    f"Rent mismatch for {gitter_id}: got {result_rent}, expected {expected_rent}"
    
    def test_no_wucher_violations(self, campaign_data_with_ids, rent_data_with_ids):
        """Verify no wucher flags appear on squares with rent below threshold."""
        # Create rent data with some high values that will trigger wucher detection
        rent_data = rent_data_with_ids.copy()
        
        # Set specific squares with high rent for wucher detection
        high_rent_indices = [0, 5, 10]  # These will be wucher outliers
        rent_data.loc[high_rent_indices, 'durchschnMieteQM'] = 20.0
        
        # Set rest to below threshold
        other_indices = [i for i in range(len(rent_data)) if i not in high_rent_indices]
        rent_data.loc[other_indices, 'durchschnMieteQM'] = 6.0
        
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data
        }
        
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # Check: No square with rent < 8.0 should have wucher_miete_flag = True
        min_rent_threshold = WUCHER_DETECTION_PARAMS['min_rent_threshold']
        violations = result[
            (result['wucher_miete_flag'] == True) & 
            (result['durchschnMieteQM'] < min_rent_threshold)
        ]
        
        assert len(violations) == 0, \
            f"Found {len(violations)} violations: wucher=True with rent < {min_rent_threshold}"
        
        if len(violations) > 0:
            print(f"\nViolations found:")
            print(violations[['GITTER_ID_100m', 'durchschnMieteQM', 'wucher_miete_flag']])
    
    def test_wucher_flags_assigned_correctly(self, campaign_data_with_ids):
        """Verify wucher flags are assigned to the correct squares."""
        # Create controlled rent data
        rent_data = campaign_data_with_ids.copy()
        
        # Most squares have normal rent
        rent_data['durchschnMieteQM'] = 10.0
        rent_data['AnzahlWohnungen'] = 20
        
        # Specific squares with very high rent (will be wucher outliers)
        # Make sure they're surrounded by normal rent to be detected as outliers
        outlier_indices = [12]  # Middle square in 5x5 grid
        rent_data.loc[outlier_indices, 'durchschnMieteQM'] = 25.0
        
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data
        }
        
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # Check that wucher flags are only on squares with high rent
        wucher_squares = result[result['wucher_miete_flag'] == True]
        
        for idx, row in wucher_squares.iterrows():
            assert row['durchschnMieteQM'] >= WUCHER_DETECTION_PARAMS['min_rent_threshold'], \
                f"Wucher flag on square with rent {row['durchschnMieteQM']} < threshold"
    
    def test_missing_gitter_id_handling(self, campaign_data_with_ids, rent_data_with_ids):
        """Test graceful handling when GITTER_ID_100m is missing."""
        # Remove GITTER_ID from rent data
        rent_data_no_id = rent_data_with_ids.drop(columns=['GITTER_ID_100m'])
        
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data_no_id
        }
        
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # Should return data with default values, not crash
        assert 'wucher_miete_flag' in result.columns
        assert 'durchschnMieteQM' in result.columns
        assert all(result['wucher_miete_flag'] == False)
    
    def test_preserves_all_campaign_columns(self, campaign_data_with_ids, rent_data_with_ids):
        """Verify all original campaign columns are preserved."""
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data_with_ids
        }
        
        original_columns = set(campaign_data_with_ids.columns)
        result = integrate_wucher_detection(campaign_data_with_ids, loading_dict)
        
        # All original columns should be present
        for col in original_columns:
            assert col in result.columns, f"Lost column {col} during integration"
        
        # Should have added new columns
        assert 'durchschnMieteQM' in result.columns
        assert 'wucher_miete_flag' in result.columns


@pytest.mark.integration
@pytest.mark.medium
class TestWucherJoinRegression:
    """Regression tests to ensure the bug doesn't reoccur."""
    
    def test_neighbor_rent_values_not_mixed(self, grid_squares_with_ids):
        """Ensure neighboring squares' rent values don't bleed into each other."""
        # Create grid where each square has unique rent based on position
        rent_data = grid_squares_with_ids.copy()
        
        # Assign unique rent values: row*10 + col (e.g., 00, 01, 02... 10, 11, 12...)
        rents = []
        for gitter_id in rent_data['GITTER_ID_100m']:
            # Extract position from GITTER_ID (simplified)
            # This creates unique values for testing
            rents.append(hash(gitter_id) % 100 / 10.0 + 5.0)
        
        rent_data['durchschnMieteQM'] = rents
        rent_data['AnzahlWohnungen'] = 20
        
        # Create campaign data (subset of rent grid)
        campaign_data = grid_squares_with_ids.copy()
        campaign_data['central_heating_flag'] = True
        campaign_data['fossil_heating_flag'] = True
        campaign_data['fernwaerme_flag'] = False
        campaign_data['renter_flag'] = True
        
        loading_dict = {
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data
        }
        
        result = integrate_wucher_detection(campaign_data, loading_dict)
        
        # Check: Each square should have its OWN rent value, not a neighbor's
        rent_lookup = dict(zip(rent_data['GITTER_ID_100m'], rent_data['durchschnMieteQM']))
        
        mismatches = 0
        for idx, row in result.iterrows():
            gitter_id = row['GITTER_ID_100m']
            result_rent = row['durchschnMieteQM']
            expected_rent = rent_lookup[gitter_id]
            
            if pd.notna(result_rent):
                if abs(result_rent - expected_rent) > 0.01:
                    mismatches += 1
                    print(f"Mismatch: {gitter_id} has rent {result_rent}, expected {expected_rent}")
        
        assert mismatches == 0, f"Found {mismatches} squares with wrong rent values (neighbor bleed)"


@pytest.mark.integration
@pytest.mark.fast
def test_integrate_wucher_detection_with_real_params(grid_squares_with_ids):
    """Integration test using real parameters from params.py."""
    from params import WUCHER_DETECTION_PARAMS
    
    # Create test data
    rent_data = grid_squares_with_ids.copy()
    rent_data['durchschnMieteQM'] = np.random.uniform(8.0, 15.0, len(rent_data))
    rent_data['AnzahlWohnungen'] = 20
    
    # Add a few high outliers
    rent_data.loc[0, 'durchschnMieteQM'] = 30.0
    rent_data.loc[5, 'durchschnMieteQM'] = 28.0
    
    campaign_data = grid_squares_with_ids.copy()
    campaign_data['central_heating_flag'] = True
    campaign_data['fossil_heating_flag'] = True
    campaign_data['fernwaerme_flag'] = False
    campaign_data['renter_flag'] = True
    
    loading_dict = {
        'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter': rent_data
    }
    
    # Should work without errors
    result = integrate_wucher_detection(campaign_data, loading_dict)
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert 'wucher_miete_flag' in result.columns
    assert 'durchschnMieteQM' in result.columns
    assert len(result) == len(campaign_data)

