# City Sorting Script

## Overview
The `sort_overlapping_cities.py` script identifies cities that appear in both `output_inside` and `output_outside`, and moves the overlapping files from `output_outside` to a new directory called `output_inside_additional`.

## Purpose
When running the pipeline with different input types, some cities may appear in both:
- **output_inside**: Cities processed with detailed district data (old_data_type)
- **output_outside**: Cities processed with heating shape data (new_data_type)

This script organizes overlapping cities by moving their `output_outside` files to `output_inside_additional`, keeping only non-overlapping cities in `output_outside`.

## Usage

### For Squares Data
```bash
# Dry run (preview what will be moved)
python scripts/sort_overlapping_cities.py --data-type squares --dry-run

# Actual move
python scripts/sort_overlapping_cities.py --data-type squares
```

### For Addresses Data
```bash
# Dry run (preview what will be moved)
python scripts/sort_overlapping_cities.py --data-type addresses --dry-run

# Actual move
python scripts/sort_overlapping_cities.py --data-type addresses
```

## File Naming Patterns

### Output Inside
- Pattern: `umap_{data_type}_{City}_{number}_htwk_{type}.geojson`
- Example: `umap_squares_Aachen_87_htwk_hochburg.geojson`

### Output Outside
- Pattern: `umap_{data_type}_{City}_heating_shape.geojson`
- Example: `umap_squares_Aachen_heating_shape.geojson`

### Output Inside Additional (Destination)
- Same as Output Outside pattern
- Located at: `output/output_inside_additional/{data_type}/`

## What the Script Does

1. **Scans output_inside**: Extracts all unique city names from filenames
2. **Searches output_outside**: Finds files with matching city names
3. **Moves files**: Cuts (not copies) matching files to `output_inside_additional`
4. **Preserves non-overlapping**: Leaves cities unique to `output_outside` untouched

## Example Results (Squares)

From the initial run:
- **Cities in output_inside**: 84 unique cities
- **Overlapping cities found**: 77 cities
- **Files moved**: 77 files to `output_inside_additional/squares/`
- **Files remaining in output_outside**: 1335 non-overlapping city files

## Directory Structure

```
output/
├── output_inside/
│   ├── squares/
│   │   └── umap_squares_Berlin_75_htwk_hochburg.geojson
│   │   └── umap_squares_Berlin_76_htwk_prekaer.geojson
│   │   └── ... (multiple files per city)
│   └── addresses/
│       └── (similar structure)
├── output_outside/
│   ├── squares/
│   │   └── umap_squares_Aalen_heating_shape.geojson  (non-overlapping)
│   │   └── umap_squares_Achern_heating_shape.geojson (non-overlapping)
│   │   └── ... (1335+ non-overlapping cities)
│   └── addresses/
│       └── (similar structure)
└── output_inside_additional/
    ├── squares/
    │   └── umap_squares_Berlin_heating_shape.geojson  (overlapping, moved)
    │   └── umap_squares_Aachen_heating_shape.geojson  (overlapping, moved)
    │   └── ... (77 overlapping cities)
    └── addresses/
        └── (similar structure)
```

## Safety Features

- **Dry-run mode**: Preview changes before executing
- **Detailed logging**: Shows which files are being moved
- **Summary report**: Provides statistics on the operation
- **Error handling**: Catches and reports any file operation errors

## Notes

- The script uses `shutil.move()`, which **removes** files from the source location
- Always run with `--dry-run` first to verify the operation
- City names with special characters (ä, ö, ü, ß) are handled correctly
- The script automatically creates destination directories if they don't exist

