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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from params import (
    INPUT_FOLDER_PATH, BEZIRKE_FOLDER_PATH, OUTPUT_PATH_SQUARES, OUTPUT_PATH_ADDRESSES,
    LOG_LEVEL, CRS, THRESHOLD_PARAMS, DATASET_NAMES, WUCHER_DETECTION_PARAMS
)
from src.functions import (
    load_geojson_folder, gdf_dict_to_crs, get_rent_campaign_df,
    filter_squares_invoting_distirct, get_all_addresses, save_all_to_geojson,
    detect_wucher_miete
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
        "--detect-wucher",
        action="store_true",
        help="Enable Wucher Miete (rent gouging) detection and save results"
    )
    
    parser.add_argument(
        "--wucher-output",
        type=str,
        default="output/wucher_miete/",
        help="Output path for Wucher Miete detection results (default: output/wucher_miete/)"
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
    heating_type, energy_type, renter_df, threshold_dict: dict
):
    """Compute rent campaign analysis data."""
    logging.info("Computing rent campaign analysis")
    
    rent_campaign_df = get_rent_campaign_df(
        heating_type=heating_type,
        energy_type=energy_type, 
        renter_df=renter_df,
        threshold_dict=threshold_dict
    )
    
    logging.info(f"Rent campaign analysis complete. Result shape: {rent_campaign_df.shape}")
    return rent_campaign_df


def filter_by_districts(bezirke_dict: dict, rent_campaign_df):
    """Filter squares by voting districts."""
    logging.info("Filtering squares by voting districts")
    
    results_dict = filter_squares_invoting_distirct(bezirke_dict, rent_campaign_df)
    
    total_squares = sum(len(gdf) for gdf in results_dict.values())
    logging.info(f"Filtered results: {len(results_dict)} districts, {total_squares} total squares")
    
    return results_dict


def extract_addresses(results_dict: dict):
    """Extract addresses for filtered squares."""
    logging.info("Extracting addresses using Overpass API")
    
    addresses_results_dict = get_all_addresses(results_dict)
    
    total_addresses = sum(len(gdf) for gdf in addresses_results_dict.values())
    logging.info(f"Address extraction complete: {total_addresses} total addresses")
    
    return addresses_results_dict


def save_results(results_dict: dict, addresses_results_dict: dict, 
                output_squares: str, output_addresses: str):
    """Save analysis results to GeoJSON files."""
    logging.info("Saving results to GeoJSON files")
    
    # Ensure output directories exist
    Path(output_squares).mkdir(parents=True, exist_ok=True)
    Path(output_addresses).mkdir(parents=True, exist_ok=True)
    
    # Save squares
    save_all_to_geojson(
        results_dict, 
        base_path=output_squares,
        kind="squares"
    )
    
    # Save addresses  
    save_all_to_geojson(
        addresses_results_dict,
        base_path=output_addresses,
        kind="addresses"
    )
    
    logging.info(f"Results saved to {output_squares} and {output_addresses}")


def run_wucher_detection(loading_dict: dict, output_path: str) -> None:
    """Run Wucher Miete detection on rent data."""
    logging.info("Starting Wucher Miete (rent gouging) detection...")
    
    # Look for rent data in the loaded datasets
    rent_dataset_name = DATASET_NAMES.get("rent")
    if not rent_dataset_name:
        logging.warning("No rent dataset configured - skipping Wucher detection")
        return
    
    if rent_dataset_name not in loading_dict:
        logging.warning(f"Rent dataset '{rent_dataset_name}' not found in input data - skipping Wucher detection")
        return
    
    rent_gdf = loading_dict[rent_dataset_name]
    logging.info(f"Running Wucher detection on {len(rent_gdf):,} rent records")
    
    # Run detection with parameters from params.py
    wucher_results = detect_wucher_miete(rent_gdf, **WUCHER_DETECTION_PARAMS)
    
    if len(wucher_results) > 0:
        # Reproject to EPSG:4326 for uMap compatibility (same as squares/addresses)
        logging.info("Reprojecting Wucher results to EPSG:4326 for uMap compatibility")
        wucher_results_4326 = wucher_results.to_crs('EPSG:4326')
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_file = output_dir / "wucher_miete_outliers.geojson"
        logging.info(f"Saving {len(wucher_results_4326):,} Wucher cases to: {output_file}")
        wucher_results_4326.to_file(output_file, driver='GeoJSON')
        
        # Log summary statistics
        rent_col = WUCHER_DETECTION_PARAMS["rent_column"]
        outlier_rents = wucher_results[rent_col]
        logging.info(f"Wucher detection summary:")
        logging.info(f"  Cases detected: {len(wucher_results):,}")
        logging.info(f"  Rent range: {outlier_rents.min():.2f} - {outlier_rents.max():.2f} EUR/sqm")
        logging.info(f"  Mean rent: {outlier_rents.mean():.2f} EUR/sqm")
        logging.info(f"  Median rent: {outlier_rents.median():.2f} EUR/sqm")
        
    else:
        logging.info("No Wucher cases detected with current parameters")
    
    logging.info("Wucher Miete detection completed")


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
            heating_type, energy_type, renter_df, THRESHOLD_PARAMS
        )
        
        # Filter by districts
        results_dict = filter_by_districts(bezirke_dict, rent_campaign_df)
        
        # Extract addresses
        addresses_results_dict = extract_addresses(results_dict)
        
        # Save results
        save_results(
            results_dict, addresses_results_dict,
            args.output_squares, args.output_addresses
        )
        
        # Optional: Wucher Miete detection
        if args.detect_wucher:
            run_wucher_detection(loading_dict, args.wucher_output)
        
        logging.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
