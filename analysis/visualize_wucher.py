#!/usr/bin/env python3
"""
Quick visualization of Wucher Miete detection results.
Creates a map showing the first 1000 detected cases.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_wucher_visualization():
    """Create and save a visualization of wucher detection results."""
    
    print("Loading Wucher Miete detection results...")
    
    # Load the wucher detection results
    wucher_file = "output/wucher_miete/wucher_miete_outliers.geojson"
    if not Path(wucher_file).exists():
        print(f"❌ Error: {wucher_file} not found. Run the pipeline with --detect-wucher first.")
        return
    
    # Load and take first 1000 entries
    gdf = gpd.read_file(wucher_file)
    print(f"📊 Loaded {len(gdf):,} total wucher cases")
    
    # Take first 1000 for visualization
    gdf_sample = gdf.head(1000).copy()
    print(f"📍 Visualizing first {len(gdf_sample):,} cases")
    
    # Reproject to Web Mercator for basemap
    gdf_sample_web = gdf_sample.to_crs('EPSG:3857')
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot the points
    rent_col = 'durchschnMieteQM'
    scatter = gdf_sample_web.plot(
        ax=ax,
        column=rent_col,
        cmap='Reds',
        markersize=20,
        alpha=0.7,
        edgecolors='darkred',
        linewidth=0.5,
        legend=True,
        legend_kwds={
            'label': 'Rent (EUR/sqm)',
            'orientation': 'vertical',
            'shrink': 0.8,
            'pad': 0.05
        }
    )
    
    # Add basemap
    try:
        ctx.add_basemap(
            ax, 
            crs=gdf_sample_web.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6
        )
        print("✅ Added OpenStreetMap basemap")
    except Exception as e:
        print(f"⚠️  Could not add basemap: {e}")
        ax.set_facecolor('lightgray')
    
    # Styling
    ax.set_title(
        'Wucher Miete Detection Results\n'
        f'First 1,000 Detected Cases (of {len(gdf):,} total)\n'
        f'Rent Range: {gdf_sample[rent_col].min():.1f} - {gdf_sample[rent_col].max():.1f} EUR/sqm',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Remove axis labels (they're not meaningful in Web Mercator)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add statistics text box
    stats_text = f"""
Detection Statistics:
• Total Cases: {len(gdf):,}
• Shown: {len(gdf_sample):,}
• Min Rent: {gdf_sample[rent_col].min():.2f} EUR/sqm
• Max Rent: {gdf_sample[rent_col].max():.2f} EUR/sqm
• Mean Rent: {gdf_sample[rent_col].mean():.2f} EUR/sqm
• Median Rent: {gdf_sample[rent_col].median():.2f} EUR/sqm
    """.strip()
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure in analysis folder
    output_file = "analysis/wucher_miete_visualization.png"
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    print(f"✅ Visualization saved as: {output_file}")
    print(f"📊 Map shows rent outliers ranging from {gdf_sample[rent_col].min():.1f} to {gdf_sample[rent_col].max():.1f} EUR/sqm")
    
    # Show some sample locations
    print(f"\n📍 Sample outlier locations:")
    for i, (idx, row) in enumerate(gdf_sample.head(5).iterrows()):
        centroid = row.geometry.centroid
        print(f"  {i+1}. {row[rent_col]:.2f} EUR/sqm at ({centroid.x:.0f}, {centroid.y:.0f})")
    
    plt.close()

if __name__ == "__main__":
    create_wucher_visualization()
