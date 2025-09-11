# Analysis Scripts and Results

This folder contains analysis scripts and generated results for the rent campaign project.

## 📊 Available Analyses

### Wucher Miete (Rent Gouging) Detection

**Script:** `visualize_wucher.py`  
**Output:** `wucher_miete_visualization.png`

Creates a map visualization of detected rent gouging cases.

**Usage:**
```bash
# Run from project root
python analysis/visualize_wucher.py
```

**Features:**
- Shows first 1,000 detected cases on an interactive map
- Color-coded by rent level (darker red = higher rent)
- Includes OpenStreetMap basemap for geographic context
- Statistics box with key detection metrics
- High-resolution PNG output (300 DPI)

**Generated Files:**
- `wucher_miete_visualization.png` - Map visualization
- `wucher_results_updated_params.geojson` - Full detection results

## 🔧 Requirements

The visualization script requires:
- `contextily` - For basemap tiles
- `matplotlib` - For plotting
- `geopandas` - For geospatial data handling

Install with:
```bash
pip install contextily
```

## 📈 Adding New Analyses

To add new analysis scripts:

1. Create your script in this folder
2. Import project modules using:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
3. Save outputs to this folder
4. Update this README with usage instructions

## 📁 Folder Structure

```
analysis/
├── README.md                           # This file
├── visualize_wucher.py                 # Wucher detection visualization
├── wucher_miete_visualization.png      # Generated map
└── wucher_results_updated_params.geojson  # Detection results
```
