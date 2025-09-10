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
│   └── test_smoke.py     # Smoke tests for pipeline validation
├── params.py             # Configuration constants and parameters
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore patterns
```

## Setup

### Prerequisites

- Python 3.8+
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
   - Place census GeoJSON files in a folder (default: `data/rent_campagne/`)
   - Place district boundaries in another folder (default: `data/all_stimmbezirke/`)

## Usage

### Basic Usage

Run the pipeline with default parameters:

```bash
python scripts/pipeline.py
```

### Custom Parameters

Override configuration via command line arguments:

```bash
python scripts/pipeline.py \
  --input-folder /path/to/census/data \
  --bezirke-folder /path/to/districts \
  --output-squares output/squares/ \
  --output-addresses output/addresses/ \
  --log-level DEBUG
```

### Adjust Analysis Thresholds

Fine-tune the analysis criteria:

```bash
python scripts/pipeline.py \
  --central-heating-threshold 0.7 \
  --fossil-heating-threshold 0.6 \
  --fernwaerme-threshold 0.1 \
  --renter-threshold 0.65
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

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
