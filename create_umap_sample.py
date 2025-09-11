#!/usr/bin/env python3
"""
Create a smaller sample of Wucher Miete detection results for uMap.
Useful when the full dataset (32k+ records) crashes uMap.
"""

import geopandas as gpd
import argparse
from pathlib import Path
import os

def create_umap_sample(input_file: str, output_file: str, sample_size: int = 2000, method: str = 'first'):
    """Create a smaller sample of wucher detection results for uMap."""
    
    print(f"📂 Loading {input_file}...")
    
    # Load the full file
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found.")
        print("   Run the pipeline with --detect-wucher first.")
        return False
    
    gdf = gpd.read_file(input_file)
    print(f"📊 Loaded {len(gdf):,} total records")
    
    if len(gdf) <= sample_size:
        print(f"✅ File already has {len(gdf):,} records (≤ {sample_size}), copying as-is")
        sample_gdf = gdf.copy()
    else:
        # Create sample based on method
        if method == 'first':
            sample_gdf = gdf.head(sample_size)
            print(f"✂️  Taking first {sample_size:,} records")
        elif method == 'random':
            sample_gdf = gdf.sample(n=sample_size, random_state=42)
            print(f"🎲 Taking random {sample_size:,} records")
        elif method == 'highest':
            rent_col = 'durchschnMieteQM'
            sample_gdf = gdf.nlargest(sample_size, rent_col)
            print(f"🔥 Taking {sample_size:,} highest rent records")
        elif method == 'stratified':
            # Take samples from different rent ranges
            rent_col = 'durchschnMieteQM'
            q25, q50, q75 = gdf[rent_col].quantile([0.25, 0.5, 0.75])
            
            # Take samples from each quartile
            per_quartile = sample_size // 4
            q1 = gdf[gdf[rent_col] <= q25].sample(n=min(per_quartile, len(gdf[gdf[rent_col] <= q25])), random_state=42)
            q2 = gdf[(gdf[rent_col] > q25) & (gdf[rent_col] <= q50)].sample(n=min(per_quartile, len(gdf[(gdf[rent_col] > q25) & (gdf[rent_col] <= q50)])), random_state=42)
            q3 = gdf[(gdf[rent_col] > q50) & (gdf[rent_col] <= q75)].sample(n=min(per_quartile, len(gdf[(gdf[rent_col] > q50) & (gdf[rent_col] <= q75)])), random_state=42)
            q4 = gdf[gdf[rent_col] > q75].sample(n=min(per_quartile, len(gdf[gdf[rent_col] > q75])), random_state=42)
            
            sample_gdf = pd.concat([q1, q2, q3, q4], ignore_index=True)
            print(f"📊 Taking stratified {len(sample_gdf):,} records across rent ranges")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Show statistics of the sample
    rent_col = 'durchschnMieteQM'
    print(f"\n📈 Sample statistics:")
    print(f"   Records: {len(sample_gdf):,}")
    print(f"   Rent range: {sample_gdf[rent_col].min():.2f} - {sample_gdf[rent_col].max():.2f} EUR/sqm")
    print(f"   Mean rent: {sample_gdf[rent_col].mean():.2f} EUR/sqm")
    print(f"   Median rent: {sample_gdf[rent_col].median():.2f} EUR/sqm")
    
    # Check geographic distribution
    bounds = sample_gdf.total_bounds
    print(f"\n📏 Geographic coverage:")
    print(f"   Longitude: {bounds[0]:.2f}° to {bounds[2]:.2f}°")
    print(f"   Latitude: {bounds[1]:.2f}° to {bounds[3]:.2f}°")
    
    # Save the sample
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    sample_gdf.to_file(output_file, driver='GeoJSON')
    
    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"\n💾 Sample file created:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size / (1024*1024):.1f} MB")
    print(f"   Records: {len(sample_gdf):,}")
    print(f"   Reduction: {len(gdf):,} → {len(sample_gdf):,} ({len(sample_gdf)/len(gdf)*100:.1f}%)")
    
    print(f"\n✅ Sample ready for uMap import!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Create smaller samples of Wucher Miete detection results for uMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Methods:
  first      - Take first N records (default, fastest)
  random     - Random sample across all records
  highest    - Take N records with highest rents
  stratified - Sample evenly across rent ranges

Examples:
  python create_umap_sample.py
  python create_umap_sample.py --size 1000 --method highest
  python create_umap_sample.py --method random --output custom_sample.geojson
        """
    )
    
    parser.add_argument(
        "--input",
        default="output/wucher_miete/wucher_miete_outliers.geojson",
        help="Input wucher detection file (default: output/wucher_miete/wucher_miete_outliers.geojson)"
    )
    
    parser.add_argument(
        "--output",
        default="output/wucher_miete/wucher_miete_outliers_small.geojson",
        help="Output sample file (default: output/wucher_miete/wucher_miete_outliers_small.geojson)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Sample size (default: 2000)"
    )
    
    parser.add_argument(
        "--method",
        choices=["first", "random", "highest", "stratified"],
        default="first",
        help="Sampling method (default: first)"
    )
    
    args = parser.parse_args()
    
    success = create_umap_sample(args.input, args.output, args.size, args.method)
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.exit(main())
