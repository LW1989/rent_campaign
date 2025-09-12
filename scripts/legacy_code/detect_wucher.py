#!/usr/bin/env python3
"""
Wucher Miete (Rent Gouging) Detection Pipeline

This script detects potential rent gouging by identifying spatial outliers
in rent data using neighborhood analysis on a regular grid.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from params import LOG_LEVEL, WUCHER_DETECTION_PARAMS
from src.functions import detect_wucher_miete
import geopandas as gpd

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
    """Parse command line arguments for wucher detection."""
    parser = argparse.ArgumentParser(
        description="Wucher Miete (Rent Gouging) Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/detect_wucher.py --input data/rent_campagne/Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter.geojson
  python scripts/detect_wucher.py --input rent_data.geojson --output wucher_results.geojson --threshold 2.0
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input GeoJSON file with rent data"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/wucher_miete_outliers.geojson",
        help="Output path for detected Wucher Miete cases (default: output/wucher_miete_outliers.geojson)"
    )

    parser.add_argument(
        "--method",
        type=str,
        default=WUCHER_DETECTION_PARAMS["method"],
        choices=["mean", "median"],
        help=f"Statistical method for neighbor comparison (default: {WUCHER_DETECTION_PARAMS['method']})"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=WUCHER_DETECTION_PARAMS["threshold"],
        help=f"Number of std deviations above neighbors to flag as outlier (default: {WUCHER_DETECTION_PARAMS['threshold']})"
    )

    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=WUCHER_DETECTION_PARAMS["neighborhood_size"],
        help=f"Size of neighborhood for comparison (must be odd, default: {WUCHER_DETECTION_PARAMS['neighborhood_size']})"
    )

    parser.add_argument(
        "--min-rent-threshold",
        type=float,
        default=WUCHER_DETECTION_PARAMS["min_rent_threshold"],
        help=f"Minimum rent per sqm to consider (default: {WUCHER_DETECTION_PARAMS['min_rent_threshold']})"
    )

    parser.add_argument(
        "--rent-column",
        type=str,
        default=WUCHER_DETECTION_PARAMS["rent_column"],
        help=f"Name of rent column in input data (default: {WUCHER_DETECTION_PARAMS['rent_column']})"
    )

    return parser.parse_args()

def main() -> int:
    """Main execution for wucher detection pipeline."""
    try:
        args = parse_args()
        setup_logging(LOG_LEVEL)
        logging.info("Starting Wucher Miete detection pipeline")

        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        logging.info(f"Loading rent data from: {input_path}")
        
        # Load rent data
        rent_gdf = gpd.read_file(input_path)
        logging.info(f"Loaded {len(rent_gdf):,} rent data records")

        # Validate rent column
        if args.rent_column not in rent_gdf.columns:
            available_cols = list(rent_gdf.columns)
            raise ValueError(f"Rent column '{args.rent_column}' not found. Available columns: {available_cols}")

        # Show rent statistics
        rent_stats = rent_gdf[args.rent_column].describe()
        logging.info(f"Rent statistics (EUR/sqm): count={rent_stats['count']:,.0f}, "
                   f"mean={rent_stats['mean']:.2f}, std={rent_stats['std']:.2f}, "
                   f"min={rent_stats['min']:.2f}, max={rent_stats['max']:.2f}")

        # Run wucher detection
        logging.info("Starting Wucher Miete detection...")
        wucher_results = detect_wucher_miete(
            rent_gdf,
            rent_column=args.rent_column,
            method=args.method,
            threshold=args.threshold,
            neighborhood_size=args.neighborhood_size,
            min_rent_threshold=args.min_rent_threshold
        )

        logging.info(f"Detection completed. Found {len(wucher_results):,} potential Wucher Miete cases")

        if len(wucher_results) > 0:
            # Show statistics of detected outliers
            outlier_rents = wucher_results[args.rent_column]
            percentage = (len(wucher_results) / len(rent_gdf)) * 100
            
            logging.info(f"Outlier statistics:")
            logging.info(f"  Count: {len(wucher_results):,} ({percentage:.2f}% of total)")
            logging.info(f"  Rent range: {outlier_rents.min():.2f} - {outlier_rents.max():.2f} EUR/sqm")
            logging.info(f"  Mean rent: {outlier_rents.mean():.2f} EUR/sqm")
            logging.info(f"  Median rent: {outlier_rents.median():.2f} EUR/sqm")

            # Reproject to EPSG:4326 for uMap compatibility
            logging.info("Reprojecting results to EPSG:4326 for uMap compatibility")
            wucher_results_4326 = wucher_results.to_crs('EPSG:4326')

            # Prepare output directory
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results
            logging.info(f"Saving results to: {output_path}")
            wucher_results_4326.to_file(output_path, driver='GeoJSON')
            logging.info(f"✅ Results saved successfully")

            # Show sample locations
            logging.info("Sample outlier locations:")
            for i, (idx, row) in enumerate(wucher_results.head(5).iterrows()):
                centroid = row.geometry.centroid
                logging.info(f"  {i+1}. Rent: {row[args.rent_column]:.2f} EUR/sqm at "
                           f"({centroid.x:.0f}, {centroid.y:.0f})")

        else:
            logging.warning("No Wucher Miete cases detected with current parameters")
            logging.info("Consider adjusting --threshold or --min-rent-threshold parameters")

        logging.info("Wucher Miete detection pipeline completed successfully")
        return 0

    except Exception as e:
        logging.error(f"Wucher detection pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
