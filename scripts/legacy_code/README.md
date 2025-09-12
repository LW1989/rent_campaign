# Legacy Wucher Detection Scripts

This folder contains older versions of Wucher Miete detection scripts that have been replaced by improved implementations.

## Files

### `detect_wucher.py`
- **Status**: Deprecated
- **Purpose**: Original standalone Wucher Miete detection script
- **Replaced by**: Integration into main pipeline (`scripts/pipeline.py`)
- **Issue**: Created duplicate detection logic instead of reusing existing functions

### `detect_wucher_with_neighbors.py`
- **Status**: Deprecated  
- **Purpose**: Complex attempt to extract neighborhood context during detection
- **Replaced by**: `scripts/simple_wucher_neighbors.py`
- **Issue**: Overcomplicated approach trying to modify the internal detection process

## Current Active Script

The current working script for Wucher detection with neighborhood context is:
- **`scripts/simple_wucher_neighbors.py`** - Simple, reliable approach that:
  1. Uses existing `detect_wucher_miete()` function
  2. Finds neighbors using simple distance calculation
  3. Creates uMap-ready GeoJSON files

## Usage

These legacy scripts are kept for reference only. Do not use them in production.
For current Wucher detection, use:

```bash
# Detect outliers with neighborhood context for uMap
python scripts/simple_wucher_neighbors.py --sample 50

# Standard detection integrated in main pipeline  
python scripts/pipeline.py
```
