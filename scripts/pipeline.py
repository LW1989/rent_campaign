#!/usr/bin/env python3
"""
Rent Campaign Analysis Pipeline

Main pipeline script that processes census and geographic data to identify
rental campaign targets based on heating type, energy source, and renter demographics.
"""

import argparse
import logging
import sys
from pathlib import Path
import geopandas as gpd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from params import (
    INPUT_FOLDER_PATH, BEZIRKE_FOLDER_PATH, OUTPUT_PATH_SQUARES, OUTPUT_PATH_ADDRESSES,
    LOG_LEVEL, CRS, THRESHOLD_PARAMS, DATASET_NAMES, WUCHER_DETECTION_PARAMS,
    CONVERSATION_STARTERS, METRIC_CARD_CONFIG
)
from src.functions import (
    load_geojson_folder, gdf_dict_to_crs, get_rent_campaign_df,
    filter_squares_invoting_distirct, get_all_addresses, save_all_to_geojson,
    detect_wucher_miete, add_conversation_starters,
    load_city_boundaries, map_districts_to_cities, calculate_city_means,
    add_metric_cards_to_districts
)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rent Campaign Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py
  python scripts/pipeline.py --input-folder /path/to/data --output-squares /path/to/squares/
  python scripts/pipeline.py --bezirke-folder data/auswahl_theo --skip-addresses  # Fast mode: squares only
        """
    )
    
    parser.add_argument(
        "--input-folder", 
        type=str, 
        default=INPUT_FOLDER_PATH,
        help=f"Path to input GeoJSON folder (default: {INPUT_FOLDER_PATH})"
    )
    
    parser.add_argument(
        "--bezirke-folder",
        type=str,
        default=BEZIRKE_FOLDER_PATH, 
        help=f"Path to districts (Bezirke) GeoJSON folder (default: {BEZIRKE_FOLDER_PATH})"
    )
    
    parser.add_argument(
        "--output-squares",
        type=str,
        default=OUTPUT_PATH_SQUARES,
        help=f"Output path for squares GeoJSON files (default: {OUTPUT_PATH_SQUARES})"
    )
    
    parser.add_argument(
        "--output-addresses",
        type=str,
        default=OUTPUT_PATH_ADDRESSES,
        help=f"Output path for addresses GeoJSON files (default: {OUTPUT_PATH_ADDRESSES})"
    )
    
    parser.add_argument(
        "--skip-addresses",
        action="store_true",
        help="Skip address extraction (fastest, only generates squares)"
    )
    
    parser.add_argument(
        "--selection-type",
        type=str,
        choices=["old_selection", "new_selection"],
        default="old_selection",
        help="Data selection type for color coding (old_selection=red, new_selection=grey)"
    )
    
    
    return parser.parse_args()


def load_input_data(input_folder: str) -> dict:
    """Load all input GeoJSON files."""
    logging.info(f"Loading input data from: {input_folder}")
    loading_dict = load_geojson_folder(input_folder)
    
    if not loading_dict:
        raise ValueError(f"No GeoJSON files found in {input_folder}")
    
    logging.info(f"Loaded {len(loading_dict)} datasets")
    return loading_dict


def load_district_data(bezirke_folder: str, crs: str) -> dict:
    """Load and transform district data."""
    logging.info(f"Loading district data from: {bezirke_folder}")
    bezirke_dict = load_geojson_folder(bezirke_folder)
    
    if not bezirke_dict:
        raise ValueError(f"No GeoJSON files found in {bezirke_folder}")
    
    # Transform to working CRS
    logging.info(f"Transforming districts to CRS: {crs}")
    bezirke_dict = gdf_dict_to_crs(bezirke_dict, crs)
    
    logging.info(f"Loaded {len(bezirke_dict)} district datasets")
    return bezirke_dict


def extract_datasets(loading_dict: dict) -> tuple:
    """Extract required datasets from loaded data."""
    try:
        renter_df = loading_dict[DATASET_NAMES["renter"]].copy()
        heating_type = loading_dict[DATASET_NAMES["heating_type"]].copy()
        energy_type = loading_dict[DATASET_NAMES["energy_type"]].copy()
        
        logging.info("Successfully extracted required datasets")
        return renter_df, heating_type, energy_type
        
    except KeyError as e:
        available_keys = list(loading_dict.keys())
        raise ValueError(
            f"Required dataset not found: {e}. "
            f"Available datasets: {available_keys}"
        )


def compute_rent_campaign_data(
    heating_type, energy_type, renter_df, threshold_dict: dict, loading_dict: dict = None
):
    """Compute rent campaign analysis data."""
    logging.info("Computing rent campaign analysis")
    
    # Create default heating and energy type share lists and labels based on actual data columns
    heating_typeshare_list = ['Fernheizung_share', 'Etagenheizung_share', 'Blockheizung_share', 'Zentralheizung_share', 'Einzel_Mehrraumoefen_share']
    energy_type_share_list = ['fossil_heating_share', 'renewable_share', 'fernwaerme_share']
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
    
    rent_campaign_df = get_rent_campaign_df(
        heating_type=heating_type,
        energy_type=energy_type, 
        renter_df=renter_df,
        heating_typeshare_list=heating_typeshare_list,
        energy_type_share_list=energy_type_share_list,
        heating_labels=heating_labels,
        energy_labels=energy_labels,
        threshold_dict=threshold_dict
    )
    
    logging.info(f"Rent campaign analysis complete. Result shape: {rent_campaign_df.shape}")
    return rent_campaign_df


def load_demographics_data(loading_dict: dict):
    """
    Extract demographics data from already loaded data dictionary.
    
    Parameters
    ----------
    loading_dict : dict
        Dictionary containing already loaded GeoDataFrames
        
    Returns
    -------
    gpd.GeoDataFrame or None
        Demographics GeoDataFrame if found, None otherwise
    """
    try:
        logging.info("Extracting demographics data from loaded datasets")
        demographics_gdf = loading_dict.get("demographics")
        if demographics_gdf is not None:
            logging.info(f"Found demographics data with {len(demographics_gdf):,} records")
            logging.debug(f"Demographics columns: {list(demographics_gdf.columns)}")
        else:
            logging.warning("Demographics data not found in loaded datasets")
        return demographics_gdf
    except Exception as e:
        logging.error(f"Error extracting demographics data: {e}")
        return None


def integrate_demographics_data(rent_campaign_df, demographics_gdf):
    """
    Integrate demographics data into rent campaign dataframe.
    
    Parameters
    ----------
    rent_campaign_df : gpd.GeoDataFrame
        Main rent campaign analysis results
    demographics_gdf : gpd.GeoDataFrame or None
        Demographics data to integrate
        
    Returns
    -------
    gpd.GeoDataFrame
        Enhanced rent campaign dataframe with demographics data
    """
    if demographics_gdf is None:
        logging.warning("No demographics data to integrate")
        return rent_campaign_df
    
    logging.info("Integrating demographics data into rent campaign analysis")
    
    try:
        # Use GITTER_ID_100m for merging if available, otherwise use geometry
        merge_col = "GITTER_ID_100m" if "GITTER_ID_100m" in rent_campaign_df.columns else "geometry"
        logging.debug(f"Using '{merge_col}' column for merging demographics data")
        
        # Check if merge column exists in demographics data
        if merge_col not in demographics_gdf.columns:
            logging.error(f"Merge column '{merge_col}' not found in demographics data. Available columns: {list(demographics_gdf.columns)}")
            logging.warning("Skipping demographics integration due to missing merge column")
            return rent_campaign_df
        
        # Perform spatial merge with demographics data
        result = rent_campaign_df.merge(
            demographics_gdf.drop(columns=['geometry']),  # Drop geometry to avoid conflicts
            on=merge_col,
            how='left'
        )
        
        # Fill missing demographics values with 0
        demographics_cols = ['AnteilUeber65', 'AnteilAuslaender', 'durchschnFlaechejeBew', 'Einwohner']
        for col in demographics_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        logging.info(f"Demographics integration complete. Result shape: {result.shape}")
        logging.debug(f"Added demographics columns: {demographics_cols}")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to integrate demographics data: {e}")
        logging.warning("Continuing without demographics data")
        return rent_campaign_df


def integrate_wucher_detection(rent_campaign_df, loading_dict: dict):
    """
    Integrate Wucher Miete detection into rent campaign data.
    
    This function runs Wucher detection on rent data and performs a LEFT JOIN
    with the rent campaign dataframe to preserve all squares while adding
    wucher_miete_flag and rent information.
    
    Parameters
    ----------
    rent_campaign_df : gpd.GeoDataFrame
        Main rent campaign analysis results with flags
    loading_dict : dict
        Dictionary containing all loaded datasets including rent data
        
    Returns
    -------
    gpd.GeoDataFrame
        Enhanced rent campaign dataframe with wucher_miete_flag and durchschnMieteQM columns
    """
    logging.info("Integrating Wucher Miete detection into rent campaign data")
    
    try:
        # Get rent dataset name and check if available
        rent_dataset_name = DATASET_NAMES.get("rent")
        if not rent_dataset_name or rent_dataset_name not in loading_dict:
            logging.warning(f"Rent dataset '{rent_dataset_name}' not found - adding default wucher_miete_flag=False")
            rent_campaign_df['wucher_miete_flag'] = False
            rent_campaign_df['durchschnMieteQM'] = None
            return rent_campaign_df
        
        rent_gdf = loading_dict[rent_dataset_name]
        logging.info(f"Running Wucher detection on {len(rent_gdf):,} rent records")
        
        # Run Wucher detection to get outliers
        wucher_outliers = detect_wucher_miete(rent_gdf, **WUCHER_DETECTION_PARAMS)
        logging.info(f"Detected {len(wucher_outliers):,} Wucher Miete outliers")
        
        # Add wucher flag to outliers
        if len(wucher_outliers) > 0:
            wucher_outliers['wucher_miete_flag'] = True
        
        # Prepare rent data for joining (add rent column to all rent squares)
        rent_col = WUCHER_DETECTION_PARAMS['rent_column']
        rent_for_join = rent_gdf[[rent_col, 'geometry']].copy()
        
        # First: LEFT JOIN rent campaign with rent data to get rent values
        logging.info("Joining rent campaign data with rent information")
        # Use spatial join to match squares
        campaign_with_rent = rent_campaign_df.sjoin(
            rent_for_join, 
            how='left', 
            predicate='intersects'
        )
        
        # Clean up join artifacts and handle duplicates
        if 'index_right' in campaign_with_rent.columns:
            campaign_with_rent = campaign_with_rent.drop('index_right', axis=1)
        
        # Remove duplicates keeping first match
        campaign_with_rent = campaign_with_rent.drop_duplicates(subset=['geometry'])
        
        # Second: LEFT JOIN with Wucher outliers to add wucher flags
        if len(wucher_outliers) > 0:
            logging.info("Joining with Wucher Miete outliers")
            # Prepare outliers for join (only need flag and geometry)
            outliers_for_join = wucher_outliers[['wucher_miete_flag', 'geometry']].copy()
            
            # Spatial join with outliers
            final_result = campaign_with_rent.sjoin(
                outliers_for_join,
                how='left',
                predicate='intersects'
            )
            
            # Clean up join artifacts
            if 'index_right' in final_result.columns:
                final_result = final_result.drop('index_right', axis=1)
            
            # Remove duplicates
            final_result = final_result.drop_duplicates(subset=['geometry'])
        else:
            final_result = campaign_with_rent.copy()
    
        # Set default values for missing wucher flags
        if 'wucher_miete_flag' not in final_result.columns:
            final_result['wucher_miete_flag'] = False
        else:
            final_result['wucher_miete_flag'] = final_result['wucher_miete_flag'].fillna(False)
        
        # Ensure boolean type for flag
        final_result['wucher_miete_flag'] = final_result['wucher_miete_flag'].astype(bool)
        
        # Clean up rent column if missing
        if rent_col not in final_result.columns:
            final_result[rent_col] = None
        
        logging.info(f"Integration complete. Final shape: {final_result.shape}")
        logging.info(f"Wucher cases in final data: {final_result['wucher_miete_flag'].sum():,}")
        logging.info(f"Squares with rent data: {final_result[rent_col].notna().sum():,}")
        
        return final_result
        
    except Exception as e:
        logging.error(f"Failed to integrate Wucher Miete detection: {e}")
        logging.warning("Continuing without Wucher Miete detection - adding default values")
        rent_campaign_df['wucher_miete_flag'] = False
        rent_campaign_df['durchschnMieteQM'] = None
        return rent_campaign_df


def filter_by_districts(bezirke_dict: dict, rent_campaign_df):
    """Filter squares by voting districts and extract colors."""
    logging.info("Filtering squares by voting districts")
    
    results_dict, color_dict = filter_squares_invoting_distirct(bezirke_dict, rent_campaign_df)
    
    total_squares = sum(len(gdf) for gdf in results_dict.values())
    logging.info(f"Filtered results: {len(results_dict)} districts, {total_squares} total squares")
    logging.info(f"Extracted colors for {len([c for c in color_dict.values() if c])} districts")
    
    return results_dict, color_dict


def add_metric_cards(results_dict: dict, rent_campaign_df):
    """Add metric cards to district results."""
    logging.info("Adding metric cards to district results")
    
    try:
        # Load city boundaries
        krs_gdf = load_city_boundaries()
        
        # Map districts to cities
        district_city_mapping = map_districts_to_cities(results_dict, krs_gdf)
        
        # Get metric columns from config
        metric_columns = list(METRIC_CARD_CONFIG.keys())
        
        # Calculate city means
        city_means = calculate_city_means(rent_campaign_df, krs_gdf, metric_columns)
        
        # Add metric cards to districts
        enhanced_results = add_metric_cards_to_districts(
            results_dict, district_city_mapping, city_means, METRIC_CARD_CONFIG
        )
        
        logging.info("Metric cards integration complete")
        return enhanced_results
        
    except Exception as e:
        logging.error(f"Failed to add metric cards: {e}")
        logging.warning("Continuing without metric cards")
        return results_dict


def extract_addresses(results_dict: dict):
    """Extract addresses for filtered squares."""
    logging.info("Extracting addresses using Overpass API")
    
    addresses_results_dict = get_all_addresses(results_dict)
    
    total_addresses = sum(len(gdf) for gdf in addresses_results_dict.values())
    logging.info(f"Address extraction complete: {total_addresses} total addresses")
    
    return addresses_results_dict


def save_results(results_dict: dict, addresses_results_dict: dict, 
                output_squares: str, output_addresses: str, selection_type: str = "old_selection",
                district_colors: dict = None):
    """Save analysis results to GeoJSON files."""
    logging.info("Saving results to GeoJSON files")
    
    # Ensure output directories exist
    Path(output_squares).mkdir(parents=True, exist_ok=True)
    Path(output_addresses).mkdir(parents=True, exist_ok=True)
    
    # Add conversation starters to squares before saving
    logging.info("Adding conversation starters to squares")
    enhanced_results_dict = {}
    for district_name, gdf in results_dict.items():
        try:
            if gdf.empty:
                logging.warning(f"Skipping empty squares for district {district_name}")
                enhanced_results_dict[district_name] = gdf
                continue
            enhanced_gdf = add_conversation_starters(gdf, CONVERSATION_STARTERS)
            enhanced_results_dict[district_name] = enhanced_gdf
            logging.debug(f"Added conversation starters to {district_name}: {len(enhanced_gdf)} squares")
        except Exception as e:
            logging.error(f"Failed to add conversation starters to squares for district {district_name}: {e}")
            enhanced_results_dict[district_name] = gdf  # Keep original data
            continue
    
    # Add conversation starters to addresses before saving (if addresses were extracted)
    enhanced_addresses_results_dict = {}
    if addresses_results_dict:
        logging.info("Adding conversation starters to addresses")
        for district_name, gdf in addresses_results_dict.items():
            try:
                if gdf.empty:
                    logging.warning(f"Skipping empty addresses for district {district_name}")
                    enhanced_addresses_results_dict[district_name] = gdf
                    continue
                enhanced_gdf = add_conversation_starters(gdf, CONVERSATION_STARTERS)
                enhanced_addresses_results_dict[district_name] = enhanced_gdf
                logging.debug(f"Added conversation starters to {district_name}: {len(enhanced_gdf)} addresses")
            except Exception as e:
                logging.error(f"Failed to add conversation starters to addresses for district {district_name}: {e}")
                enhanced_addresses_results_dict[district_name] = gdf  # Keep original data
                continue
    else:
        logging.info("Skipping address export (no addresses extracted)")
    
    # Define columns to exclude from final output
    exclude_columns = ['description', 'renter_flag', 'tooltip', 'flag_key']
    
    # Save enhanced squares (excluding specified columns)
    save_all_to_geojson(
        enhanced_results_dict, 
        base_path=output_squares,
        kind="squares",
        exclude_cols=exclude_columns,
        selection_type=selection_type,
        district_colors=district_colors
    )
    
    # Save enhanced addresses (excluding specified columns) only if addresses were extracted
    if enhanced_addresses_results_dict:
        save_all_to_geojson(
            enhanced_addresses_results_dict,
            base_path=output_addresses,
            kind="addresses",
            exclude_cols=exclude_columns,
            selection_type=selection_type,
            district_colors=district_colors
        )
        logging.info(f"Results saved to {output_squares} and {output_addresses}")
    else:
        logging.info(f"Squares saved to {output_squares} (addresses skipped)")




def main() -> int:
    """Main pipeline execution."""
    try:
        # Parse arguments (only paths)
        args = parse_args()
        
        # Setup logging from params
        setup_logging(LOG_LEVEL)
        logging.info("Starting rent campaign analysis pipeline")
        
        # Use threshold dictionary from params
        logging.info(f"Using thresholds: {THRESHOLD_PARAMS}")
        
        # Load input data
        loading_dict = load_input_data(args.input_folder)
        
        # Load district data
        bezirke_dict = load_district_data(args.bezirke_folder, CRS)
        
        # Extract required datasets
        renter_df, heating_type, energy_type = extract_datasets(loading_dict)
        
        # Compute rent campaign analysis
        rent_campaign_df = compute_rent_campaign_data(
            heating_type, energy_type, renter_df, THRESHOLD_PARAMS, loading_dict
        )
        
        # Extract demographics data from already loaded data
        demographics_gdf = load_demographics_data(loading_dict)
        
        # Integrate demographics data
        rent_campaign_df = integrate_demographics_data(rent_campaign_df, demographics_gdf)
        
        # Integrate Wucher Miete detection
        rent_campaign_df = integrate_wucher_detection(rent_campaign_df, loading_dict)
        
        # Filter by districts and extract colors
        results_dict, district_colors = filter_by_districts(bezirke_dict, rent_campaign_df)
        
        # Add metric cards to district results
        results_dict = add_metric_cards(results_dict, rent_campaign_df)
        
        # Extract addresses (optional)
        if args.skip_addresses:
            logging.info("Skipping address extraction (--skip-addresses flag set)")
            addresses_results_dict = {}
        else:
            addresses_results_dict = extract_addresses(results_dict)
        
        # Save results
        save_results(
            results_dict, addresses_results_dict,
            args.output_squares, args.output_addresses,
            selection_type=args.selection_type,
            district_colors=district_colors
        )
        
        logging.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
