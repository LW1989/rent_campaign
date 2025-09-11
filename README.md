# Rent Campaign Analysis

A geospatial analysis pipeline for identifying rental campaign targets based on census data, heating types, energy sources, and renter demographics.

## Purpose

This repository processes German census data (Zensus 2022) to identify geographic areas suitable for rental campaigns by analyzing:

- **Central heating prevalence**: Areas with high central heating adoption
- **Fossil fuel dependency**: Areas heavily reliant on gas, oil, or coal heating
- **District heating availability**: Areas with low district heating (Fernwärme) coverage  
- **Renter demographics**: Areas with high renter populations

The pipeline generates GeoJSON files with both geographic squares and individual addresses for campaign targeting.

## Repository Structure

```
rent_campaign/
├── src/
│   ├── __init__.py
│   └── functions.py      # Core analysis functions with type hints and docstrings
├── scripts/
│   ├── __init__.py  
│   └── pipeline.py       # Main pipeline script with CLI interface
├── tests/
│   ├── unit/             # Unit tests for individual functions
│   ├── integration/      # Integration tests for workflows
│   ├── performance/      # Performance and memory tests
│   ├── data/            # Test data and fixtures
│   ├── utils/           # Test utilities and factories
│   └── conftest.py      # Shared pytest fixtures
├── params.py             # Configuration constants and parameters
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore patterns
```

## Setup

### Prerequisites

- Python 3.9+
- pip or conda for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rent_campaign
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**

   **Option A: Use preprocessed GeoJSON files (recommended)**
   - Place census GeoJSON files in `data/rent_campagne/`
   - Place district boundaries in `data/all_stimmbezirke/`
   
   **Option B: Start with raw Zensus 2022 CSV files**
   - Place raw CSV files in `data/raw_csv/`
   - Run preprocessing: `python scripts/preprocess.py`
   - This will create GeoJSON files directly in `data/rent_campagne/`

## Usage

### Data Preprocessing (if using raw CSV files)

If you have raw Zensus 2022 CSV files, run the preprocessing pipeline first:

```bash
# Basic preprocessing (data/raw_csv → data/rent_campagne)
python scripts/preprocess.py

# Custom input/output paths
python scripts/preprocess.py \
  --input-path /path/to/csv/files \
  --output-path /path/to/processed/geojson

# Different output format
python scripts/preprocess.py --output-format gpkg
```

**What the preprocessing does:**
- Converts CSV files to GeoDataFrames with 100m×100m grid polygons
- Handles German decimal format (comma → period conversion)
- Drops unnecessary columns (coordinates, explanatory text)
- Creates proper geometries from `GITTER_ID_100m` grid identifiers
- Saves as GeoJSON files ready for the main pipeline

### Wucher Miete (Rent Gouging) Detection

Detect potential rent gouging by identifying spatial outliers in rent data:

```bash
# Basic detection on preprocessed rent data
python scripts/detect_wucher.py \
  --input data/rent_campagne/Durchschnittliche_Nettokaltmiete_und_Anzahl_der_Wohnungen_100m-Gitter.geojson

# Customize detection parameters
python scripts/detect_wucher.py \
  --input rent_data.geojson \
  --output wucher_results.geojson \
  --threshold 2.0 \
  --method median \
  --neighborhood-size 5 \
  --min-rent-threshold 3.0
```

**What the wucher detection does:**
- Uses spatial neighborhood analysis to identify rent outliers
- Compares each grid cell's rent to its local neighbors (3×3, 5×5, or 7×7 neighborhood)
- Flags cells with rents significantly above neighborhood median/mean
- Configurable sensitivity via threshold parameter (lower = more sensitive)
- Outputs GeoJSON file with detected outlier locations and statistics
- Filters out low/invalid rents to focus on genuine outliers

### Main Analysis Pipeline

Run the main analysis pipeline:

```bash
python scripts/pipeline.py
```

### Custom File Paths

Override input and output paths via command line arguments:

```bash
python scripts/pipeline.py \
  --input-folder /path/to/census/data \
  --bezirke-folder /path/to/districts \
  --output-squares output/squares/ \
  --output-addresses output/addresses/
```

### Adjust Analysis Parameters

To modify analysis thresholds, logging level, or other settings, edit the `params.py` file:

```python
# Edit params.py
THRESHOLD_PARAMS = {
    "central_heating_thres": 0.7,    # Increase central heating threshold
    "fossil_heating_thres": 0.5,     # Decrease fossil fuel threshold  
    "fernwaerme_thres": 0.1,         # Lower district heating threshold
    "renter_share": 0.65             # Adjust renter threshold
}

LOG_LEVEL = "DEBUG"  # Change logging verbosity
```

### View All Options

```bash
python scripts/pipeline.py --help
```

## Configuration

Edit `params.py` to modify default settings:

```python
# Analysis thresholds
THRESHOLD_PARAMS = {
    "central_heating_thres": 0.6,    # 60% central heating threshold
    "fossil_heating_thres": 0.6,     # 60% fossil fuel threshold  
    "fernwaerme_thres": 0.2,         # 20% district heating threshold
    "renter_share": 0.6              # 60% renter threshold
}

# Input/output paths
INPUT_FOLDER_PATH = "data/rent_campagne"
OUTPUT_PATH_SQUARES = "output/squares/"
```

## Expected Data Format

The pipeline expects GeoJSON files with specific naming conventions:

### Required Input Files

- **Renter data**: `Zensus2022_Eigentuemerquote_100m-Gitter.geojson`
  - Must contain `Eigentuemerquote` column (ownership percentage)

- **Heating type data**: `Zensus2022_Gebaeude_mit_Wohnraum_nach_ueberwiegender_Heizungsart_100m-Gitter.geojson`
  - Must contain heating type columns: `Fernheizung`, `Etagenheizung`, `Blockheizung`, `Zentralheizung`, etc.

- **Energy source data**: `Zensus2022_Gebaeude_mit_Wohnraum_nach_Energietraeger_der_Heizung_100m-Gitter.geojson`
  - Must contain energy type columns: `Gas`, `Heizoel`, `Kohle`, `Fernwaerme`, etc.

### District Boundaries

Any GeoJSON files in the districts folder representing administrative boundaries.

## Output

The pipeline generates two types of GeoJSON output:

### 1. Squares (`output/squares/`)
Geographic grid squares meeting the analysis criteria, with metadata:
- `central_heating_flag`: Boolean flag for central heating prevalence
- `fossil_heating_flag`: Boolean flag for fossil fuel dependency
- `fernwaerme_flag`: Boolean flag for district heating availability
- `renter_flag`: Boolean flag for renter demographics
- `district_name`: Administrative district name

### 2. Addresses (`output/addresses/`)
Individual address points within qualifying squares, extracted via Overpass API:
- Address components (`street`, `housenumber`, `postcode`, `city`)
- Same boolean flags as squares
- uMap-ready tooltips and styling

## Testing

Run the smoke tests to verify the pipeline works correctly:

```bash
python -m pytest tests/test_smoke.py -v
```

Or run with the unittest module:

```bash
python tests/test_smoke.py
```

The tests validate:
- Core analysis functions with synthetic data
- Data processing pipeline integration
- GeoJSON file creation and loading
- Threshold calculations and flag generation

## Dependencies

Core libraries:
- **geopandas**: Geospatial data processing
- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computing
- **shapely**: Geometric operations
- **matplotlib**: Data visualization
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **requests**: HTTP requests for Overpass API
- **loguru**: Advanced logging

See `requirements.txt` for specific version requirements.

## Troubleshooting

### Common Issues

1. **Missing input files**: Ensure GeoJSON files follow the expected naming convention
2. **CRS mismatches**: All input files should have consistent coordinate reference systems
3. **Memory issues**: Large datasets may require processing in chunks
4. **Overpass API timeouts**: Address extraction may fail for very large areas

### Logging

Increase log verbosity for debugging:

```bash
python scripts/pipeline.py --log-level DEBUG
```

### Performance

For large datasets:
- Consider processing districts individually
- Adjust `min_overlap_ratio` in spatial parameters
- Monitor memory usage during address extraction

## Testing

The project includes a comprehensive test suite with multiple test categories:

### Test Categories

- **Fast tests** (`pytest -m fast`): Unit tests that run in < 30 seconds
- **Medium tests** (`pytest -m medium`): Integration tests that run in < 5 minutes  
- **Slow tests** (`pytest -m slow`): End-to-end tests that run in < 30 minutes
- **Performance tests** (`pytest -m performance`): Benchmarks and scaling tests
- **Data quality tests** (`pytest -m data_quality`): Real data validation

### Running Tests

1. **Run all fast tests:**
   ```bash
   pytest -m fast
   ```

2. **Run with coverage:**
   ```bash
   pytest --cov=src --cov-report=html
   ```

3. **Run specific test file:**
   ```bash
   pytest tests/unit/test_core_functions.py -v
   ```

4. **Run parameterized tests:**
   ```bash
   pytest tests/unit/test_wucher_parameterized.py -v
   ```

5. **Validate test infrastructure:**
   ```bash
   python test_validation.py
   ```

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual functions
│   ├── test_core_functions.py
│   ├── test_wucher_detection.py
│   └── test_wucher_parameterized.py
├── integration/          # Integration and workflow tests
│   ├── test_pipeline.py
│   └── test_data_quality.py
├── performance/          # Performance and memory tests
│   ├── test_benchmarks.py
│   └── test_memory.py
├── utils/               # Test utilities and data factories
│   └── factories.py
└── conftest.py          # Shared fixtures and test configuration
```

### CI/CD

The project includes GitHub Actions workflows for:

- **Continuous Integration**: Runs on every push/PR with fast and integration tests
- **Pull Request Validation**: Comprehensive validation for PRs including performance checks
- **Nightly Tests**: Extended test suite including slow tests and memory leak detection

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
