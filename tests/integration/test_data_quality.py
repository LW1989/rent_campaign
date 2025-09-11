#!/usr/bin/env python3
"""
Data quality validation tests for real CSV files.

These tests validate the structure and quality of actual data files
used in the rent campaign pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import re


@pytest.mark.data_quality
@pytest.mark.medium
class TestRealDataQuality:
    """Test the quality and structure of real CSV data files."""
    
    def test_csv_files_exist(self, real_csv_files):
        """Test that expected CSV files exist."""
        csv_names = [f.name for f in real_csv_files]
        
        expected_files = [
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter.csv',
            'Zensus2022_Eigentuemerquote_100m-Gitter.csv'
        ]
        
        for expected_file in expected_files:
            assert expected_file in csv_names, f"Missing expected CSV file: {expected_file}"
    
    def test_rent_csv_structure(self, real_csv_files):
        """Test the structure of the rent CSV file."""
        rent_file = None
        for csv_file in real_csv_files:
            if 'Nettokaltmiete' in csv_file.name:
                rent_file = csv_file
                break
        
        if rent_file is None:
            pytest.skip("Rent CSV file not found")
        
        # Read first few rows to check structure
        df = pd.read_csv(rent_file, sep=';', nrows=1000)
        
        # Check required columns
        required_columns = [
            'GITTER_ID_100m',
            'x_mp_100m', 
            'y_mp_100m',
            'durchschnMieteQM',
            'AnzahlWohnungen',
            'werterlaeuternde_Zeichen'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check GITTER_ID format
        gitter_pattern = r'^CRS3035RES100mN\d+E\d+$'
        valid_gitter_ids = df['GITTER_ID_100m'].str.match(gitter_pattern)
        
        assert valid_gitter_ids.all(), "Some GITTER_ID values don't match expected pattern"
        
        # Check coordinate ranges (should be in EPSG:3035 range)
        assert df['x_mp_100m'].min() > 4000000, "X coordinates seem too small for EPSG:3035"
        assert df['x_mp_100m'].max() < 5000000, "X coordinates seem too large for EPSG:3035"
        assert df['y_mp_100m'].min() > 2500000, "Y coordinates seem too small for EPSG:3035"
        assert df['y_mp_100m'].max() < 3500000, "Y coordinates seem too large for EPSG:3035"
    
    def test_rent_values_realistic(self, real_csv_files):
        """Test that rent values are in realistic ranges."""
        rent_file = None
        for csv_file in real_csv_files:
            if 'Nettokaltmiete' in csv_file.name:
                rent_file = csv_file
                break
        
        if rent_file is None:
            pytest.skip("Rent CSV file not found")
        
        # Read sample of data
        df = pd.read_csv(rent_file, sep=';', nrows=10000)
        
        # Convert German decimal format to float
        df['rent_numeric'] = df['durchschnMieteQM'].astype(str).str.replace(',', '.').astype(float)
        
        # Check rent ranges
        assert df['rent_numeric'].min() >= 0, "Found negative rent values"
        assert df['rent_numeric'].max() <= 200, "Found extremely high rent values (>200 EUR/sqm)"
        
        # Check that most rents are in reasonable range (2-50 EUR/sqm)
        reasonable_rents = df['rent_numeric'].between(2.0, 50.0)
        reasonable_percentage = reasonable_rents.mean()
        
        assert reasonable_percentage > 0.8, f"Only {reasonable_percentage:.1%} of rents are in reasonable range (2-50 EUR/sqm)"
        
        # Check apartment counts
        assert df['AnzahlWohnungen'].min() >= 1, "Found apartment counts < 1"
        assert df['AnzahlWohnungen'].max() <= 1000, "Found extremely high apartment counts"
    
    def test_ownership_csv_structure(self, real_csv_files):
        """Test the structure of the ownership CSV file."""
        ownership_file = None
        for csv_file in real_csv_files:
            if 'Eigentuemerquote' in csv_file.name:
                ownership_file = csv_file
                break
        
        if ownership_file is None:
            pytest.skip("Ownership CSV file not found")
        
        # Read first few rows
        df = pd.read_csv(ownership_file, sep=';', nrows=1000)
        
        # Check required columns
        required_columns = [
            'GITTER_ID_100m',
            'x_mp_100m',
            'y_mp_100m', 
            'Eigentuemerquote',
            'werterlaeuternde_Zeichen'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check ownership values
        ownership_values = df['Eigentuemerquote'].astype(str)
        
        # Should have missing values (–) and percentage values
        has_missing = (ownership_values == '–').any()
        has_percentages = ownership_values.str.contains(r'^\d+,\d+$', na=False).any()
        
        assert has_missing, "No missing ownership values found (expected some '–' values)"
        assert has_percentages, "No percentage ownership values found"
    
    def test_gitter_id_consistency(self, real_csv_files):
        """Test that GITTER_ID values are consistent across files."""
        if len(real_csv_files) < 2:
            pytest.skip("Need at least 2 CSV files to test consistency")
        
        gitter_ids_by_file = {}
        
        for csv_file in real_csv_files[:2]:  # Test first 2 files
            df = pd.read_csv(csv_file, sep=';', nrows=5000)
            gitter_ids_by_file[csv_file.name] = set(df['GITTER_ID_100m'])
        
        # Check that files have some overlapping GITTER_IDs
        file_names = list(gitter_ids_by_file.keys())
        overlap = gitter_ids_by_file[file_names[0]] & gitter_ids_by_file[file_names[1]]
        
        overlap_percentage = len(overlap) / len(gitter_ids_by_file[file_names[0]])
        
        assert overlap_percentage > 0.1, f"Files have very low GITTER_ID overlap: {overlap_percentage:.1%}"
    
    def test_coordinate_grid_alignment(self, real_csv_files):
        """Test that coordinates align to 100m grid."""
        rent_file = None
        for csv_file in real_csv_files:
            if 'Nettokaltmiete' in csv_file.name:
                rent_file = csv_file
                break
        
        if rent_file is None:
            pytest.skip("Rent CSV file not found")
        
        df = pd.read_csv(rent_file, sep=';', nrows=1000)
        
        # Check that coordinates end in 50 (center of 100m grid cells)
        x_centers = df['x_mp_100m'] % 100
        y_centers = df['y_mp_100m'] % 100
        
        # All should be 50 (center of grid cell)
        assert (x_centers == 50).all(), "X coordinates don't align to 100m grid centers"
        assert (y_centers == 50).all(), "Y coordinates don't align to 100m grid centers"
    
    def test_special_value_handling(self, real_csv_files):
        """Test handling of special values like KLAMMERN."""
        for csv_file in real_csv_files:
            df = pd.read_csv(csv_file, sep=';', nrows=5000)
            
            if 'werterlaeuternde_Zeichen' in df.columns:
                special_values = df['werterlaeuternde_Zeichen'].dropna()
                
                # Should contain KLAMMERN values
                has_klammern = (special_values == 'KLAMMERN').any()
                
                if len(special_values) > 0:
                    # If there are special values, KLAMMERN should be common
                    klammern_ratio = (special_values == 'KLAMMERN').mean()
                    assert klammern_ratio > 0.5, f"KLAMMERN should be most common special value, got ratio: {klammern_ratio:.2f}"


@pytest.mark.data_quality  
@pytest.mark.fast
class TestProcessedDataQuality:
    """Test the quality of processed GeoJSON data."""
    
    def test_processed_geojson_exists(self, real_data_paths):
        """Test that processed GeoJSON files exist."""
        processed_dir = real_data_paths['processed_dir']
        
        if not processed_dir.exists():
            pytest.skip("Processed data directory doesn't exist")
        
        expected_files = [
            'Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter.geojson'
        ]
        
        for expected_file in expected_files:
            file_path = processed_dir / expected_file
            assert file_path.exists(), f"Missing processed file: {expected_file}"
    
    def test_rent_geojson_sample_structure(self, sample_real_rent_data):
        """Test structure of processed rent GeoJSON data."""
        sample_gdf = sample_real_rent_data(100)  # Load small sample
        
        if sample_gdf is None:
            pytest.skip("Processed rent data not available")
        
        # Check required columns
        required_columns = ['durchschnMieteQM', 'AnzahlWohnungen', 'geometry']
        for col in required_columns:
            assert col in sample_gdf.columns, f"Missing column in processed data: {col}"
        
        # Check CRS
        assert sample_gdf.crs is not None, "GeoDataFrame missing CRS"
        assert 'EPSG:3035' in str(sample_gdf.crs), f"Unexpected CRS: {sample_gdf.crs}"
        
        # Check geometries
        assert sample_gdf.geometry.isna().sum() == 0, "Found null geometries"
        assert sample_gdf.geometry.is_valid.all(), "Found invalid geometries"
        
        # Check rent values
        rent_values = sample_gdf['durchschnMieteQM']
        assert rent_values.min() >= 0, "Found negative rent values in processed data"
        assert rent_values.max() <= 200, "Found extremely high rents in processed data"
        
        # Check apartment counts
        apt_counts = sample_gdf['AnzahlWohnungen']
        assert apt_counts.min() >= 1, "Found invalid apartment counts in processed data"
    
    def test_wucher_output_structure(self, real_data_paths):
        """Test structure of Wucher detection output."""
        wucher_file = real_data_paths['wucher_output']
        
        if not wucher_file.exists():
            pytest.skip("Wucher detection output not found")
        
        import geopandas as gpd
        
        # Load small sample
        gdf = gpd.read_file(wucher_file, rows=100)
        
        # Check required columns
        required_columns = ['durchschnMieteQM', 'AnzahlWohnungen', 'wucher_miete_flag', 'geometry']
        for col in required_columns:
            assert col in gdf.columns, f"Missing column in wucher output: {col}"
        
        # Check that all entries are flagged as outliers
        assert gdf['wucher_miete_flag'].all(), "Not all entries in wucher output are flagged as outliers"
        
        # Check CRS (should be EPSG:4326 for uMap compatibility)
        assert 'EPSG:4326' in str(gdf.crs) or 'WGS84' in str(gdf.crs), f"Wucher output has wrong CRS: {gdf.crs}"
        
        # Check rent values are above threshold
        rent_values = gdf['durchschnMieteQM']
        assert rent_values.min() >= 6.0, "Found wucher outliers below minimum rent threshold"
