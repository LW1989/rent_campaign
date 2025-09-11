# Test Migration Notice

## Old vs New Test Structure

The original `test_smoke.py` file has been replaced with a modern, comprehensive test suite.

### Migration Summary

**Old Structure:**
```
tests/
└── test_smoke.py  # Single file with all tests (626 lines)
```

**New Structure:**
```
tests/
├── unit/                           # Unit tests
│   ├── test_core_functions.py      # Core pipeline functions
│   ├── test_wucher_detection.py    # Wucher detection functions
│   ├── test_wucher_parameterized.py # Parameterized Wucher tests
│   └── test_hypothesis_properties.py # Property-based tests
├── integration/                    # Integration tests
│   ├── test_pipeline.py           # Pipeline workflows
│   └── test_data_quality.py       # Real data validation
├── performance/                    # Performance tests
│   ├── test_benchmarks.py         # Performance benchmarks
│   └── test_memory.py             # Memory and leak tests
├── utils/                         # Test utilities
│   └── factories.py              # Test data factories
└── conftest.py                   # Shared fixtures
```

### What Was Migrated

✅ **All original test functionality** preserved and enhanced
✅ **Converted from unittest to pytest** format
✅ **Added realistic test data** based on actual CSV structure
✅ **Enhanced with parameterized testing** for better coverage
✅ **Added property-based testing** with Hypothesis
✅ **Added performance and memory testing**
✅ **Added data quality validation**

### Running Migrated Tests

**Old way:**
```bash
python -m unittest tests.test_smoke
```

**New way:**
```bash
# Run all tests
pytest

# Run by category
pytest -m fast          # Quick unit tests
pytest -m medium         # Integration tests  
pytest -m performance    # Performance tests

# Run specific components
pytest tests/unit/test_core_functions.py
pytest tests/unit/test_wucher_detection.py
```

### Key Improvements

1. **Better Organization**: Tests grouped by functionality and speed
2. **Realistic Data**: Test fixtures based on actual CSV formats and values
3. **Enhanced Coverage**: Parameterized and property-based testing
4. **Performance Monitoring**: Benchmarks and memory leak detection
5. **CI/CD Integration**: Automated testing with GitHub Actions

### Backwards Compatibility

The old `test_smoke.py` file is preserved for reference but **should not be used**. All functionality has been migrated to the new structure with improvements.

To fully migrate:
1. Use the new test commands shown above
2. Update any scripts that referenced `test_smoke.py`
3. Follow the new test organization for future test additions

### Questions?

If you need to find where a specific test was migrated to:
- **Core function tests** → `tests/unit/test_core_functions.py`
- **Preprocessing tests** → `tests/integration/test_pipeline.py`
- **Wucher detection tests** → `tests/unit/test_wucher_detection.py`
- **Pipeline integration** → `tests/integration/test_pipeline.py`
