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
    "energy_type": "Zensus2022_Gebaeude_mit_Wohnraum_nach_Energietraeger_der_Heizung_100m-Gitter",
    "rent": "Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter"
}

# Demographics dataset file mapping (without extensions)
DEMOGRAPHICS_DATASETS = {
    "elderly_share": "Zensus2022_Anteil_ueber_65_100m-Gitter",
    "foreigner_share": "Zensus2022_Anteil_Auslaender_100m-Gitter", 
    "area_per_person": "Zensus2022_Durchschn_Flaeche_je_Bewohner_100m-Gitter",
    "population": "Zensus2022_Bevoelkerungszahl_100m-Gitter"
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

# Wucher Miete (rent gouging) detection parameters
WUCHER_DETECTION_PARAMS = {
    "method": "median",          # 'mean' or 'median' for neighbor statistic
    "threshold": 3,            # Number of standard deviations above neighbor median/mean
    "neighborhood_size": 11,      # Size of neighborhood (must be odd: 3, 5, 7, etc.)
    "min_rent_threshold": 8.0,   # Minimum rent per sqm to consider (filter out very low rents)
    "min_neighbors": 30,          # Minimum number of neighbors required for comparison
    "rent_column": "durchschnMieteQM"  # Column name for rent per square meter
}


# Metric card configuration
METRIC_CARD_CONFIG = {
    "durchschnMieteQM": {
        "id": "rent_per_m2",
        "label": "Miete €/m²"
    },
    "AnteilUeber65": {
        "id": "elderly_share",
        "label": "Anteil über 65"
    },
    "AnteilAuslaender": {
        "id": "foreigner_share", 
        "label": "Anteil Ausländer"
    },
    "durchschnFlaechejeBew": {
        "id": "area_per_person",
        "label": "Fläche je Bewohner"
    },
    "Einwohner": {
        "id": "population",
        "label": "Einwohner"
    }
}

# Direct MultiPolygon Format Configuration
DIRECT_MULTIPOLYGON_CONFIG = {
    'format_name': 'heide_multipolygon',
    'expected_geometry_type': 'MultiPolygon',
    'expected_feature_count': 1,
    'polygon_count_range': (1, 20),  # Expected range of polygons per MultiPolygon
    'naming_pattern': r'^\d+\.\d+$',  # Numeric pattern like "39.0"
    'processing_mode': 'direct_multipolygon',  # Use MultiPolygon directly
    'name_parsing': 'flexible',  # Handle both "|" and non-"|" formats
    'extraction_required': False,  # No geometry extraction needed
}

# Name Parsing Configuration
NAME_PARSING_CONFIG = {
    'stimmbezirke_pattern': r'.*\|\s*([^|]+)\s*\|.*',  # Extract middle part
    'numeric_pattern': r'^\d+\.\d+$',  # Heide/Neumünster format
    'fallback_behavior': 'use_full_name',  # Use full name if no pattern matches
}

# Opacity scaling configuration for population-based visualization
OPACITY_SCALING_CONFIG = {
    "enabled": True,                    # Feature flag to enable/disable
    "min_opacity": 0.1,                 # Minimum fillOpacity (sparse areas = faint)
    "max_opacity": 0.9,                 # Maximum fillOpacity (dense areas = bold)
    "fallback_opacity": 0.9,           # Default opacity when scaling unavailable
    "population_column": "Einwohner",   # Column name for population data
    
    # Outlier handling with percentile-based clipping
    "use_robust_scaling": True,         # Enable percentile-based clipping
    "lower_percentile": 5,              # Lower clip boundary (%)
    "upper_percentile": 95,             # Upper clip boundary (%)
}

# conversation starters

CONVERSATION_STARTERS = {
    "0000": "Hallo, ich bin von der Linken. Wir reden gerade mit den Nachbar*innen darüber, wie es ihnen mit den Wohn- und Nebenkosten geht. Wie erleben Sie das hier?",
    "0001": "Viele hier im Viertel erzählen uns, dass die Miete kaum noch zu stemmen ist. Wie geht es Ihnen damit – passt das noch oder wird's eng?",
    "0010": "Manche Nachbar*innen berichten, dass sie mit der Fernwärme schwer durchblicken bei den Kosten. Wie ist das bei Ihnen – alles nachvollziehbar oder eher undurchsichtig?",
    "0011": "Ich höre hier oft: Fernwärme ist teuer, und dazu kommen hohe Mieten. Wie wirkt sich das bei Ihnen aus – ist das für Sie ein Thema?",
    "0100": "Die Preise für Gas und Öl sind zuletzt enorm gestiegen. Haben Sie das bei sich auch gemerkt, oder hält es sich bei Ihnen noch in Grenzen?",
    "0101": "Viele zahlen gerade für fossile Heizung und eine hohe Miete – das summiert sich schnell. Wie erleben Sie das bei sich zu Hause?",
    "1000": "Zentralheizungen können praktisch sein – aber oft gibt es Ärger mit der Abrechnung. Wie läuft das bei Ihnen, passt das oder eher nicht?",
    "1001": "Bei Zentralheizung und steigenden Mieten hören wir oft, dass die Gesamtkosten explodieren. Wie empfinden Sie das – geht's noch oder wird es knapp?",
    "1100": "Bei zentralen Gas- oder Ölheizungen schlagen die Energiepreise besonders durch. Spüren Sie das auch auf der Rechnung?",
    "1101": "Manche sagen: Fossile Zentralheizung und hohe Miete, das frisst richtig Geld. Wie erleben Sie das bei sich?",
    "0110": "Energiepreise sind für viele hier eine Belastung – ob Fernwärme oder Gas. Wie ist das bei Ihnen: Haben Sie den Eindruck, dass die Kosten fair verteilt sind?",

}