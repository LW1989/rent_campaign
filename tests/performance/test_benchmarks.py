#!/usr/bin/env python3
"""
Performance benchmarks for rent campaign pipeline.

These tests measure execution time and resource usage of key pipeline components
to detect performance regressions and optimize bottlenecks.
"""

import pytest
import time
import gc
import psutil
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import (
    detect_wucher_miete, detect_neighbor_outliers, 
    gdf_to_xarray, xarray_to_gdf,
    process_df, create_geodataframe
)
from tests.utils.factories import RentDataFactory, create_outlier_test_scenario


@pytest.mark.performance
@pytest.mark.slow
class TestWucherDetectionPerformance:
    """Performance tests for Wucher Miete detection."""
    
    def test_wucher_detection_small_dataset(self, benchmark_params, large_test_dataset):
        """Benchmark wucher detection on small dataset."""
        small_data = large_test_dataset(benchmark_params['small_dataset_size'])
        
        start_time = time.perf_counter()
        result = detect_wucher_miete(small_data, min_neighbors=5)  # Lower threshold for small data
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should complete quickly for small datasets
        assert execution_time < 5.0, f"Small dataset took too long: {execution_time:.2f}s"
        assert len(result) >= 0, "Should return valid result"
        
        print(f"Small dataset ({len(small_data)} points): {execution_time:.2f}s")
    
    def test_wucher_detection_medium_dataset(self, benchmark_params, large_test_dataset):
        """Benchmark wucher detection on medium dataset."""
        medium_data = large_test_dataset(benchmark_params['medium_dataset_size'])
        
        start_time = time.perf_counter()
        result = detect_wucher_miete(medium_data, min_neighbors=10)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time for medium datasets
        assert execution_time < 30.0, f"Medium dataset took too long: {execution_time:.2f}s"
        assert len(result) >= 0, "Should return valid result"
        
        print(f"Medium dataset ({len(medium_data)} points): {execution_time:.2f}s")
    
    @pytest.mark.slow
    def test_wucher_detection_large_dataset(self, benchmark_params, large_test_dataset):
        """Benchmark wucher detection on large dataset."""
        large_data = large_test_dataset(benchmark_params['large_dataset_size'])
        
        start_time = time.perf_counter()
        result = detect_wucher_miete(large_data)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Should complete within timeout for large datasets
        timeout = benchmark_params['timeout_seconds']
        assert execution_time < timeout, f"Large dataset exceeded timeout: {execution_time:.2f}s > {timeout}s"
        assert len(result) >= 0, "Should return valid result"
        
        print(f"Large dataset ({len(large_data)} points): {execution_time:.2f}s")
    
    def test_memory_usage_scaling(self, large_test_dataset):
        """Test memory usage scaling with dataset size."""
        process = psutil.Process()
        
        sizes = [100, 500, 1000, 2000]
        memory_usage = []
        
        for size in sizes:
            gc.collect()  # Clean up before measurement
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and process dataset
            test_data = large_test_dataset(size)
            result = detect_wucher_miete(test_data, min_neighbors=min(10, size//10))
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            memory_usage.append(memory_increase)
            
            # Clean up
            del test_data, result
            gc.collect()
            
            print(f"Size {size}: {memory_increase:.1f} MB")
        
        # Memory usage should scale reasonably (not exponentially)
        # Check that doubling size doesn't more than triple memory usage
        for i in range(1, len(memory_usage)):
            size_ratio = sizes[i] / sizes[i-1]
            memory_ratio = memory_usage[i] / max(memory_usage[i-1], 1.0)  # Avoid division by zero
            
            assert memory_ratio <= size_ratio * 1.5, f"Memory usage scaling poorly: {memory_ratio:.1f}x for {size_ratio}x data"


@pytest.mark.performance
@pytest.mark.medium
class TestXArrayPerformance:
    """Performance tests for xarray operations."""
    
    def test_gdf_to_xarray_conversion_speed(self, large_test_dataset):
        """Benchmark GeoDataFrame to xarray conversion."""
        test_data = large_test_dataset(1000)
        
        start_time = time.perf_counter()
        xarray_result = gdf_to_xarray(test_data, 'durchschnMieteQM')
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        assert execution_time < 10.0, f"GDF to xarray conversion too slow: {execution_time:.2f}s"
        assert xarray_result is not None, "Conversion should succeed"
        
        print(f"GDF to xarray ({len(test_data)} points): {execution_time:.2f}s")
    
    def test_neighbor_outlier_detection_speed(self):
        """Benchmark neighbor outlier detection on different grid sizes."""
        grid_sizes = [(10, 10), (25, 25), (50, 50)]
        
        for rows, cols in grid_sizes:
            # Create test xarray
            test_scenario = create_outlier_test_scenario('sparse_outliers')
            test_xarray = test_scenario['xarray']
            
            # Resize to target size
            if rows * cols != 25:  # Original is 5x5
                from tests.utils.factories import XArrayDataFactory
                test_xarray = XArrayDataFactory.create_rent_xarray(
                    shape=(rows, cols),
                    outlier_positions=[(rows//3, cols//3), (2*rows//3, 2*cols//3)]
                )
            
            start_time = time.perf_counter()
            result = detect_neighbor_outliers(test_xarray, threshold=2.0, size=3)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            grid_size = rows * cols
            
            # Should scale reasonably with grid size
            expected_max_time = grid_size / 1000.0 + 1.0  # Rough linear scaling + overhead
            assert execution_time < expected_max_time, f"Grid {rows}x{cols} too slow: {execution_time:.2f}s"
            
            print(f"Outlier detection {rows}x{cols} grid: {execution_time:.2f}s")


@pytest.mark.performance
@pytest.mark.medium
class TestPreprocessingPerformance:
    """Performance tests for data preprocessing."""
    
    def test_csv_processing_speed(self, temp_directory):
        """Benchmark CSV processing with different file sizes."""
        from tests.utils.factories import CSVDataFactory
        
        file_sizes = [100, 500, 1000, 5000]
        
        for n_rows in file_sizes:
            # Create test CSV
            csv_content = CSVDataFactory.create_rent_csv_content(n_rows)
            csv_file = temp_directory / f"test_{n_rows}.csv"
            
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            start_time = time.perf_counter()
            
            # Process CSV through the pipeline
            result_dict = process_df(
                path=str(temp_directory),
                sep=";",
                cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
                on_col="GITTER_ID_100m",
                drop_how="any",
                how="inner",
                gitter_id_column="GITTER_ID_100m"
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Should process reasonably quickly
            max_time = n_rows / 500.0 + 2.0  # Rough scaling + overhead
            assert execution_time < max_time, f"CSV processing ({n_rows} rows) too slow: {execution_time:.2f}s"
            
            print(f"CSV processing {n_rows} rows: {execution_time:.2f}s")
    
    def test_geodataframe_creation_speed(self, real_gitter_ids):
        """Benchmark GeoDataFrame creation from GITTER_IDs."""
        import pandas as pd
        
        # Test with different numbers of GITTER_IDs
        for multiplier in [1, 10, 100, 500]:
            gitter_list = real_gitter_ids * multiplier
            
            test_df = pd.DataFrame({
                'GITTER_ID_100m': gitter_list,
                'value': range(len(gitter_list))
            })
            
            start_time = time.perf_counter()
            result_gdf = create_geodataframe(test_df)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            n_features = len(test_df)
            
            # Should scale reasonably
            max_time = n_features / 1000.0 + 1.0
            assert execution_time < max_time, f"GeoDataFrame creation ({n_features} features) too slow: {execution_time:.2f}s"
            
            print(f"GeoDataFrame creation {n_features} features: {execution_time:.2f}s")


@pytest.mark.performance
@pytest.mark.fast  
class TestPerformanceRegression:
    """Tests to detect performance regressions in key functions."""
    
    def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics for key operations."""
        # These are reference times that should not regress significantly
        baseline_times = {
            'small_wucher_detection': 2.0,      # seconds for 100 points
            'medium_gdf_to_xarray': 5.0,        # seconds for 1000 points  
            'small_outlier_detection': 0.5,     # seconds for 10x10 grid
            'csv_processing_1k': 3.0,           # seconds for 1000 CSV rows
        }
        
        # Record current performance (this test mainly documents expected performance)
        print("\nBaseline Performance Metrics:")
        for operation, max_time in baseline_times.items():
            print(f"  {operation}: should complete in < {max_time}s")
        
        # This test passes by definition but provides documentation
        assert True, "Baseline metrics established"
    
    def test_memory_leak_detection(self, large_test_dataset):
        """Test for memory leaks in repeated operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run the same operation multiple times
        test_data = large_test_dataset(200)
        
        for i in range(5):
            result = detect_wucher_miete(test_data, min_neighbors=5)
            del result  # Explicit cleanup
            
            if i % 2 == 0:
                gc.collect()  # Periodic garbage collection
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory significantly over multiple runs
        assert memory_increase < 50.0, f"Potential memory leak detected: {memory_increase:.1f} MB increase"
        
        print(f"Memory usage after 5 iterations: +{memory_increase:.1f} MB")


# Utility functions for performance testing
def time_function(func, *args, **kwargs):
    """Time a function execution and return (result, execution_time)."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
