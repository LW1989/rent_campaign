#!/usr/bin/env python3
"""
Zensus 2022 Data Preprocessing Pipeline

This script preprocesses raw Zensus 2022 CSV files into GeoJSON format for use
in the rent campaign analysis pipeline. It handles:

- Importing multiple CSV files from a directory
- Cleaning data (dropping columns, converting German decimal format)
- Creating geometries from GITTER_ID_100m grid coordinates
- Saving processed data as GeoJSON files

The main pipeline expects the output of this preprocessing step.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from params import PREPROCESS_PARAMS, LOG_LEVEL, DEMOGRAPHICS_DATASETS
from src.functions import process_df, save_geodataframes, process_demographics, save_geodataframe


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the preprocessing pipeline."""
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
        description="Zensus 2022 Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess.py
  python scripts/preprocess.py --input-path /path/to/csv/files --output-path /path/to/geojson/output
        """
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        default=PREPROCESS_PARAMS["input_path"],
        help=f"Path to directory containing Zensus CSV files (default: {PREPROCESS_PARAMS['input_path']})"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=PREPROCESS_PARAMS["output_path"],
        help=f"Output directory for processed GeoJSON files (default: {PREPROCESS_PARAMS['output_path']})"
    )
    
    parser.add_argument(
        "--csv-separator",
        type=str,
        default=PREPROCESS_PARAMS["csv_separator"],
        help=f"CSV delimiter character (default: '{PREPROCESS_PARAMS['csv_separator']}')"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["geojson", "gpkg", "shp"],
        default=PREPROCESS_PARAMS["output_format"],
        help=f"Output file format (default: {PREPROCESS_PARAMS['output_format']})"
    )
    
    parser.add_argument(
        "--process-demographics",
        action="store_true",
        help="Process demographics data and create merged demographics GeoJSON file"
    )
    
    parser.add_argument(
        "--demographics-output",
        type=str,
        default="demographics.geojson",
        help="Output filename for demographics data (default: demographics.geojson)"
    )
    
    return parser.parse_args()


def validate_input_path(input_path: str) -> None:
    """Validate that input path exists and contains CSV files."""
    path = Path(input_path)
    
    if not path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in input path: {input_path}")
    
    logging.info(f"Found {len(csv_files)} CSV files in {input_path}")
    for csv_file in csv_files:
        logging.debug(f"  - {csv_file.name}")


def validate_output_path(output_path: str) -> None:
    """Validate and create output directory if needed."""
    path = Path(output_path)
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory ready: {output_path}")
    except Exception as e:
        raise ValueError(f"Cannot create output directory {output_path}: {e}")


def run_preprocessing(
    input_path: str,
    output_path: str,
    csv_separator: str,
    output_format: str,
    process_demographics_flag: bool = False,
    demographics_output_filename: str = "demographics.geojson"
) -> None:
    """Run the main preprocessing workflow."""
    
    logging.info("Starting Zensus 2022 data preprocessing")
    
    # Validate paths
    validate_input_path(input_path)
    validate_output_path(output_path)
    
    # Process the CSV files
    logging.info("Processing CSV files...")
    df_dict = process_df(
        path=input_path,
        sep=csv_separator,
        cols_to_drop=PREPROCESS_PARAMS["columns_to_drop"],
        on_col=PREPROCESS_PARAMS["gitter_id_column"],
        drop_how="any",  # Not used in current implementation but required parameter
        how="inner",     # Not used in current implementation but required parameter
        gitter_id_column=PREPROCESS_PARAMS["gitter_id_column"]
    )
    
    logging.info(f"Successfully processed {len(df_dict)} datasets")
    
    # Save processed data
    logging.info(f"Saving data as {output_format.upper()} files...")
    save_geodataframes(
        gdf_dict=df_dict,
        output_dir=output_path,
        file_format=output_format
    )
    
    # Process demographics if requested
    if process_demographics_flag:
        logging.info("Processing demographics data...")
        try:
            demographics_gdf = process_demographics(
                path=input_path,
                sep=csv_separator,
                cols_to_drop=PREPROCESS_PARAMS["columns_to_drop"],
                gitter_id_column=PREPROCESS_PARAMS["gitter_id_column"],
                demographics_datasets=DEMOGRAPHICS_DATASETS
            )
            
            # Save demographics data
            demographics_output_path = Path(output_path) / demographics_output_filename
            save_geodataframe(
                gdf=demographics_gdf,
                output_path=str(demographics_output_path),
                file_format=output_format
            )
            
            logging.info(f"Demographics processing complete. Saved to: {demographics_output_path}")
            
        except Exception as e:
            logging.error(f"Demographics processing failed: {e}")
            raise
    
    logging.info(f"Preprocessing complete. Output saved to: {output_path}")


def main() -> int:
    """Main preprocessing execution."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(LOG_LEVEL)
        logging.info("Starting Zensus 2022 preprocessing pipeline")
        
        # Run preprocessing
        run_preprocessing(
            input_path=args.input_path,
            output_path=args.output_path,
            csv_separator=args.csv_separator,
            output_format=args.output_format,
            process_demographics_flag=args.process_demographics,
            demographics_output_filename=args.demographics_output
        )
        
        logging.info("Preprocessing pipeline completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
