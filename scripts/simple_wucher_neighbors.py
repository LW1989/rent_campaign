#!/usr/bin/env python3
"""
SIMPLE approach: Detect Wucher Miete and extract neighbors.

This script uses the simplest possible approach:
1. Run normal Wucher detection to get outliers
2. For each outlier, find its neighbors in the original grid
3. Export both as separate GeoJSON files for uMap

Usage:
    python scripts/simple_wucher_neighbors.py --sample 50
"""

import sys
import argparse
from pathlib import Path
import logging
import geopandas as gpd
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.functions import detect_wucher_miete, load_geojson_folder
from params import WUCHER_DETECTION_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple Wucher detection with neighbors for uMap"
    )
    
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Sample N outliers for testing (default: all)'
    )
    
    parser.add_argument(
        '--neighborhood-size', '-n',
        type=int,
        default=None,
        help=f'Override neighborhood size (default: {WUCHER_DETECTION_PARAMS["neighborhood_size"]})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/simple_wucher',
        help='Output directory (default: output/simple_wucher)'
    )
    
    return parser.parse_args()


def find_neighbors_simple(outlier_centroid, all_rent_data, neighborhood_size):
    """
    Find neighbors around an outlier using simple distance calculation.
    
    For a 100m grid, neighborhood_size=5 means we want a 5x5 area around the outlier.
    """
    
    # Calculate search radius (in meters)
    radius = (neighborhood_size * 100) / 2  # e.g., 5*100/2 = 250m radius
    
    # Get outlier coordinates
    outlier_x = outlier_centroid.x
    outlier_y = outlier_centroid.y
    
    # Find all squares within radius
    distances = all_rent_data.geometry.centroid.distance(outlier_centroid)
    neighbors_mask = distances <= radius
    
    # Exclude the outlier itself (distance < 50m = same square)
    not_self_mask = distances > 50
    
    final_mask = neighbors_mask & not_self_mask
    neighbors = all_rent_data[final_mask]
    
    return neighbors


def create_umap_files(outliers_gdf, all_neighbors_gdf, output_dir):
    """Create uMap-ready GeoJSON files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rent_col = WUCHER_DETECTION_PARAMS['rent_column']
    
    # Prepare outliers for uMap
    outliers_umap = outliers_gdf.copy()
    if outliers_umap.crs.to_epsg() != 4326:
        outliers_umap = outliers_umap.to_crs('EPSG:4326')
    
    outliers_umap['name'] = outliers_umap.apply(
        lambda row: f"🔴 Wucher Miete: {row[rent_col]:.2f} EUR/sqm", axis=1
    )
    outliers_umap['description'] = outliers_umap.apply(
        lambda row: f"""🔴 **WUCHER MIETE DETECTED**

💰 **Rent:** {row[rent_col]:.2f} EUR/sqm
🏠 **Apartments:** {row.get('AnzahlWohnungen', 'Unknown')}
🚨 **Status:** Potential rent gouging

This square was flagged as significantly higher rent than its neighborhood.

**Detection Parameters:**
• Method: {WUCHER_DETECTION_PARAMS['method']} of neighborhood
• Threshold: {WUCHER_DETECTION_PARAMS['threshold']}σ above neighbors
• Neighborhood: {WUCHER_DETECTION_PARAMS['neighborhood_size']}×{WUCHER_DETECTION_PARAMS['neighborhood_size']} squares""", axis=1
    )
    outliers_umap['marker-color'] = '#FF0000'
    outliers_umap['marker-size'] = 'large'
    outliers_umap['stroke'] = '#000000'
    outliers_umap['stroke-width'] = 3
    
    # Save outliers
    outliers_file = output_path / "wucher_outliers_umap.geojson"
    outliers_umap.to_file(outliers_file, driver='GeoJSON')
    logger.info(f"Saved {len(outliers_umap)} outliers to: {outliers_file}")
    
    # Prepare neighbors for uMap (if any)
    if len(all_neighbors_gdf) > 0:
        neighbors_umap = all_neighbors_gdf.copy()
        if neighbors_umap.crs.to_epsg() != 4326:
            neighbors_umap = neighbors_umap.to_crs('EPSG:4326')
        
        neighbors_umap['name'] = neighbors_umap.apply(
            lambda row: f"🔵 Neighbor: {row[rent_col]:.2f} EUR/sqm", axis=1
        )
        neighbors_umap['description'] = neighbors_umap.apply(
            lambda row: f"""🔵 **NEIGHBORHOOD CONTEXT**

💰 **Rent:** {row[rent_col]:.2f} EUR/sqm
🏠 **Apartments:** {row.get('AnzahlWohnungen', 'Unknown')}
🏘️ **Role:** Neighborhood square

This square provides context for outlier detection. It was used to calculate the neighborhood statistics for nearby rent gouging detection.

**Related outlier:** {row.get('outlier_id', 'Unknown')}""", axis=1
        )
        neighbors_umap['marker-color'] = '#4A90E2'
        neighbors_umap['marker-size'] = 'medium'
        neighbors_umap['stroke'] = '#000000'
        neighbors_umap['stroke-width'] = 1
        
        # Save neighbors
        neighbors_file = output_path / "wucher_neighbors_umap.geojson"
        neighbors_umap.to_file(neighbors_file, driver='GeoJSON')
        logger.info(f"Saved {len(neighbors_umap)} neighbors to: {neighbors_file}")
        
        # Create combined file
        combined = gpd.GeoDataFrame(
            pd.concat([outliers_umap, neighbors_umap], ignore_index=True),
            crs='EPSG:4326'
        )
        combined_file = output_path / "wucher_combined_umap.geojson"
        combined.to_file(combined_file, driver='GeoJSON')
        logger.info(f"Saved combined file to: {combined_file}")
        
        return outliers_file, neighbors_file, combined_file
    else:
        logger.warning("No neighbors found!")
        return outliers_file, None, outliers_file


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("🎯 Simple Wucher detection with neighbors")
    
    try:
        # Load rent data
        logger.info("Loading rent data...")
        rent_data = load_geojson_folder('data/rent_campagne')
        
        # Find rent data
        rent_gdf = None
        for key, gdf in rent_data.items():
            if 'miete' in key.lower() or 'nettokaltmiete' in key.lower():
                rent_gdf = gdf
                logger.info(f"Using: {key} ({len(gdf):,} records)")
                break
        
        if rent_gdf is None:
            available = list(rent_data.keys())
            raise ValueError(f"No rent data found. Available: {available}")
        
        # Step 1: Detect outliers using existing function
        logger.info("Step 1: Detecting Wucher Miete outliers...")
        
        detection_params = WUCHER_DETECTION_PARAMS.copy()
        if args.neighborhood_size:
            detection_params['neighborhood_size'] = args.neighborhood_size
        
        outliers_gdf = detect_wucher_miete(rent_gdf, **detection_params)
        
        if len(outliers_gdf) == 0:
            logger.warning("No outliers detected!")
            return
        
        logger.info(f"Found {len(outliers_gdf):,} outliers")
        
        # Sample outliers if requested
        if args.sample and len(outliers_gdf) > args.sample:
            logger.info(f"Sampling {args.sample} outliers from {len(outliers_gdf):,}")
            outliers_gdf = outliers_gdf.sample(n=args.sample, random_state=42)
        
        # Step 2: Find neighbors for each outlier
        logger.info("Step 2: Finding neighbors for each outlier...")
        
        all_neighbors = []
        neighborhood_size = detection_params['neighborhood_size']
        
        for i, (idx, outlier_row) in enumerate(outliers_gdf.iterrows()):
            outlier_id = f"outlier_{i}"  # Use consistent numbering
            outlier_centroid = outlier_row.geometry.centroid
            
            # Find neighbors for this outlier
            neighbors = find_neighbors_simple(
                outlier_centroid, 
                rent_gdf, 
                neighborhood_size
            )
            
            # Add metadata to neighbors
            for _, neighbor_row in neighbors.iterrows():
                neighbor_dict = neighbor_row.to_dict()
                neighbor_dict['type'] = 'neighbor'
                neighbor_dict['outlier_id'] = outlier_id
                all_neighbors.append(neighbor_dict)
            
            logger.debug(f"Outlier {outlier_id}: found {len(neighbors)} neighbors")
        
        # Create neighbors GeoDataFrame
        if all_neighbors:
            all_neighbors_gdf = gpd.GeoDataFrame(all_neighbors, crs=rent_gdf.crs)
            
            # Remove duplicates (neighbors that belong to multiple outliers)
            unique_neighbors = all_neighbors_gdf.drop_duplicates(
                subset=['geometry'], keep='first'
            )
            
            logger.info(f"Found {len(unique_neighbors):,} unique neighbors")
        else:
            unique_neighbors = gpd.GeoDataFrame()
            logger.warning("No neighbors found!")
        
        # Add outlier metadata
        outliers_gdf_with_meta = outliers_gdf.copy()
        outliers_gdf_with_meta['type'] = 'outlier'
        outliers_gdf_with_meta['outlier_id'] = [f"outlier_{i}" for i in range(len(outliers_gdf_with_meta))]
        
        # Step 3: Create uMap files
        logger.info("Step 3: Creating uMap files...")
        
        outliers_file, neighbors_file, combined_file = create_umap_files(
            outliers_gdf_with_meta, 
            unique_neighbors, 
            args.output_dir
        )
        
        # Summary
        print(f"\n✅ SUCCESS!")
        print(f"📊 Results:")
        print(f"   🔴 Outliers: {len(outliers_gdf_with_meta):,}")
        print(f"   🔵 Neighbors: {len(unique_neighbors):,}")
        print(f"\n📁 Files created:")
        print(f"   • {outliers_file}")
        if neighbors_file:
            print(f"   • {neighbors_file}")
        print(f"   • {combined_file} ← **IMPORT THIS TO uMAP**")
        
        print(f"\n🗺️ uMap Instructions:")
        print(f"   1. Go to umap.openstreetmap.fr")
        print(f"   2. Create new map")
        print(f"   3. Import data: {combined_file}")
        print(f"   4. Red markers = outliers, Blue = neighbors")
        print(f"   5. Click markers to see rent prices and details")
        
        file_size = combined_file.stat().st_size / 1024
        print(f"\n📏 File size: {file_size:.1f} KB")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
