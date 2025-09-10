"""
Configuration parameters for rent campaign analysis.

This module contains all configurable constants and dictionaries used in the pipeline.
Update these values to change pipeline behavior without modifying code.
"""

from pathlib import Path

# Input/Output paths
INPUT_FOLDER_PATH = "data/rent_campagne"
BEZIRKE_FOLDER_PATH = "data/all_stimmbezirke"
OUTPUT_PATH_SQUARES = "output/squares/"
OUTPUT_PATH_ADDRESSES = "output/addresses/"

# Logging configuration
LOG_LEVEL = "INFO"

# Coordinate Reference System
CRS = "EPSG:3035"

# Analysis thresholds
THRESHOLD_PARAMS = {
    "central_heating_thres": 0.6,
    "fossil_heating_thres": 0.6, 
    "fernwaerme_thres": 0.2,
    "renter_share": 0.6
}

# Spatial analysis parameters
SPATIAL_PARAMS = {
    "min_overlap_ratio": 0.10,
    "work_crs": "EPSG:3035"
}

# Dataset file mapping (without extensions)
DATASET_NAMES = {
    "renter": "Zensus2022_Eigentuemerquote_100m-Gitter",
    "heating_type": "Zensus2022_Gebaeude_mit_Wohnraum_nach_ueberwiegender_Heizungsart_100m-Gitter",
    "energy_type": "Zensus2022_Gebaeude_mit_Wohnraum_nach_Energietraeger_der_Heizung_100m-Gitter"
}

# Overpass API configuration
OVERPASS_CONFIG = {
    "timeout": 180,
    "retries": 4
}

# Preprocessing configuration
PREPROCESS_PARAMS = {
    "input_path": "data/raw_csv",
    "output_path": "data/rent_campagne",
    "csv_separator": ";",
    "columns_to_drop": ["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
    "gitter_id_column": "GITTER_ID_100m",
    "output_format": "geojson"
}
