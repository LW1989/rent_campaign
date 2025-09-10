# Raw Zensus 2022 CSV Files

Place your raw Zensus 2022 CSV files here for preprocessing.

## Expected Files

The preprocessing pipeline expects CSV files with the following characteristics:

- **Format**: CSV with semicolon (`;`) separator
- **Encoding**: UTF-8 
- **Required Column**: `GITTER_ID_100m` - Grid cell identifier in format `CRS3035RES100mN{north}E{east}`

## Expected Columns

The files should contain columns like:
- `GITTER_ID_100m` - Grid cell identifier (required)
- `x_mp_100m`, `y_mp_100m` - Will be dropped (coordinates derived from GITTER_ID)
- `werterlaeuternde_Zeichen` - Will be dropped (explanatory characters)
- Various data columns (heating types, energy sources, ownership, etc.)

## Data Processing

The preprocessing pipeline will:
1. Import all CSV files from this directory
2. Drop unnecessary columns
3. Convert German decimal format (comma) to standard format (period)
4. Create 100m×100m polygon geometries from GITTER_ID_100m
5. Save as GeoJSON files for the main analysis pipeline

## Usage

```bash
# Process files in this directory (default)
python scripts/preprocess.py

# Process files from custom location
python scripts/preprocess.py --input-path /path/to/csv/files
```
