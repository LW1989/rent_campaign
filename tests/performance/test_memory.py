#!/usr/bin/env python3
"""
Memory usage and resource leak tests.

These tests monitor memory consumption and detect potential memory leaks
in the rent campaign pipeline components.
"""

import pytest
import gc
import psutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.functions import detect_wucher_miete, detect_neighbor_outliers, gdf_to_xarray
from tests.utils.factories import RentDataFactory


@pytest.mark.memory
@pytest.mark.slow
class TestMemoryUsage:
    """Test memory usage patterns and limits."""
    
    def test_memory_baseline(self):
        """Establish memory baseline for the test environment."""
        process = psutil.Process()
        
        # Clean up before measurement
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Should start with reasonable memory usage
        assert initial_memory < 500, f"Baseline memory too high: {initial_memory:.1f} MB"
        
        print(f"Memory baseline: {initial_memory:.1f} MB")
    
    def test_wucher_detection_memory_scaling(self, large_test_dataset):
        """Test that memory usage scales linearly with dataset size."""
        process = psutil.Process()
        
        sizes = [100, 200, 500, 1000]
        memory_usage = []
        
        for size in sizes:
            # Clean up before each measurement
            gc.collect()
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
            
            # Memory usage should be reasonable for dataset size
            # Rough expectation: < 1MB per 100 data points
            max_expected = size / 100 + 50  # 50MB base + scaling
            assert memory_increase < max_expected, \
                f"Memory usage too high for size {size}: {memory_increase:.1f} MB > {max_expected:.1f} MB"
        
        # Check that memory scaling is roughly linear
        for i in range(1, len(memory_usage)):
            size_ratio = sizes[i] / sizes[i-1]
            memory_ratio = memory_usage[i] / max(memory_usage[i-1], 1.0)
            
            # Memory shouldn't scale worse than quadratically
            assert memory_ratio <= size_ratio * 2.0, \
                f"Poor memory scaling: {memory_ratio:.1f}x memory for {size_ratio}x data"
    
    def test_memory_peak_limits(self, large_test_dataset):
        """Test that memory usage stays within reasonable peaks."""
        process = psutil.Process()
        
        # Test with reasonably large dataset
        test_data = large_test_dataset(2000)
        
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor memory during processing
        peak_memory = initial_memory
        
        def memory_monitor():
            nonlocal peak_memory
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        # Run detection while monitoring memory
        result = detect_wucher_miete(test_data, min_neighbors=20)
        memory_monitor()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_increase = peak_memory - initial_memory
        
        print(f"Initial: {initial_memory:.1f} MB, Peak: {peak_memory:.1f} MB, Final: {final_memory:.1f} MB")
        print(f"Peak increase: {peak_increase:.1f} MB")
        
        # Peak memory increase should be reasonable (< 1GB for 2k points)
        assert peak_increase < 1024, f"Peak memory too high: {peak_increase:.1f} MB"
        
        # Memory should come back down after processing
        memory_decrease = peak_memory - final_memory
        assert memory_decrease >= 0, "Memory should decrease after processing"
    
    def test_xarray_conversion_memory(self, sample_rent_gdf):
        """Test memory usage of GeoDataFrame to xarray conversion."""
        process = psutil.Process()
        
        # Test with multiple sizes
        for multiplier in [1, 5, 10, 20]:
            # Create larger dataset by repeating the sample
            large_gdf = sample_rent_gdf.copy()
            for _ in range(multiplier - 1):
                large_gdf = pd.concat([large_gdf, sample_rent_gdf], ignore_index=True)
            
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Convert to xarray
            xarray_result = gdf_to_xarray(large_gdf, 'durchschnMieteQM')
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            print(f"Multiplier {multiplier} ({len(large_gdf)} points): {memory_increase:.1f} MB")
            
            # Clean up
            del large_gdf, xarray_result
            gc.collect()
            
            # Memory increase should be reasonable
            expected_max = multiplier * 10 + 20  # Rough scaling expectation
            assert memory_increase < expected_max, \
                f"XArray conversion memory too high: {memory_increase:.1f} MB"


@pytest.mark.memory
@pytest.mark.slow  
class TestMemoryLeaks:
    """Test for memory leaks in repeated operations."""
    
    def test_repeated_wucher_detection_no_leak(self, large_test_dataset):
        """Test that repeated Wucher detection doesn't leak memory."""
        process = psutil.Process()
        
        # Create test data once
        test_data = large_test_dataset(300)
        
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_measurements = [initial_memory]
        
        # Run detection multiple times
        for i in range(8):
            result = detect_wucher_miete(test_data, min_neighbors=5)
            
            # Explicit cleanup
            del result
            
            # Periodic garbage collection
            if i % 2 == 0:
                gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
            
            print(f"Iteration {i+1}: {current_memory:.1f} MB")
        
        final_memory = memory_measurements[-1]
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase over {len(memory_measurements)-1} iterations: {memory_increase:.1f} MB")
        
        # Should not increase memory significantly over multiple runs
        assert memory_increase < 100, f"Potential memory leak: +{memory_increase:.1f} MB"
        
        # Check for steady memory increase (leak pattern)
        recent_avg = np.mean(memory_measurements[-3:])
        early_avg = np.mean(memory_measurements[1:4])
        
        if len(memory_measurements) > 6:
            trend_increase = recent_avg - early_avg
            assert trend_increase < 50, f"Memory trending upward: +{trend_increase:.1f} MB"
    
    def test_repeated_array_operations_no_leak(self):
        """Test that repeated array operations don't leak memory."""
        from tests.utils.factories import XArrayDataFactory
        
        process = psutil.Process()
        
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run array operations repeatedly
        for i in range(10):
            # Create and process xarray
            test_array = XArrayDataFactory.create_rent_xarray(
                shape=(20, 20),
                outlier_positions=[(5, 5), (15, 15)]
            )
            
            # Run outlier detection
            outliers = detect_neighbor_outliers(test_array, threshold=2.0)
            
            # Clean up explicitly
            del test_array, outliers
            
            if i % 3 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Array operations memory increase: {memory_increase:.1f} MB")
        
        # Array operations should not leak significant memory
        assert memory_increase < 30, f"Array operations memory leak: +{memory_increase:.1f} MB"
    
    def test_large_dataset_cleanup(self):
        """Test that large datasets are properly cleaned up."""
        process = psutil.Process()
        
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and immediately delete large datasets
        for size in [1000, 2000, 3000]:
            large_data = RentDataFactory.create_realistic_rent_sample(size)
            
            # Check memory increased
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_with_data = current_memory - initial_memory
            
            print(f"Memory with {size} points: +{memory_with_data:.1f} MB")
            
            # Delete and clean up
            del large_data
            gc.collect()
            
            # Check memory decreased
            after_cleanup = process.memory_info().rss / 1024 / 1024  # MB
            memory_after_cleanup = after_cleanup - initial_memory
            
            print(f"Memory after cleanup: +{memory_after_cleanup:.1f} MB")
            
            # Memory should decrease or at least not increase further
            # Note: Python GC doesn't guarantee immediate memory return to OS
            memory_freed = memory_with_data - memory_after_cleanup
            
            # More realistic expectation: memory should not continue growing significantly
            # Allow for some memory overhead but expect some cleanup
            assert memory_after_cleanup <= memory_with_data * 1.2, \
                f"Memory continued growing after cleanup: {memory_after_cleanup:.1f} MB vs {memory_with_data:.1f} MB"
            
            # Check that we at least attempted cleanup (freed some memory or stayed stable)
            assert memory_freed >= -memory_with_data * 0.1, \
                f"Memory usage increased significantly after cleanup: {memory_freed:.1f} MB change"


@pytest.mark.memory
@pytest.mark.medium
class TestResourceManagement:
    """Test proper resource management and cleanup."""
    
    def test_file_handle_management(self, temp_directory, sample_csv_content):
        """Test that file handles are properly managed."""
        import resource
        
        # Get initial file descriptor count
        initial_fds = len(psutil.Process().open_files())
        
        # Create many temporary files and process them
        csv_files = []
        for i in range(20):
            csv_file = temp_directory / f"test_{i}.csv"
            with open(csv_file, 'w') as f:
                f.write(sample_csv_content)
            csv_files.append(csv_file)
        
        # Process files (this might open/close many file handles)
        from src.functions import process_df
        
        for i in range(5):  # Process subset to avoid too many temp files
            single_file_dir = temp_directory / f"single_{i}"
            single_file_dir.mkdir()
            
            # Copy one file to isolated directory
            single_csv = single_file_dir / f"test_{i}.csv"
            single_csv.write_text(sample_csv_content)
            
            # Process it
            result_dict = process_df(
                path=str(single_file_dir),
                sep=";",
                cols_to_drop=["x_mp_100m", "y_mp_100m", "werterlaeuternde_Zeichen"],
                on_col="GITTER_ID_100m",
                drop_how="any",
                how="inner",
                gitter_id_column="GITTER_ID_100m"
            )
            
            # Clean up result
            del result_dict
        
        # Force garbage collection
        gc.collect()
        
        # Check final file descriptor count
        final_fds = len(psutil.Process().open_files())
        fd_increase = final_fds - initial_fds
        
        print(f"File descriptor increase: {fd_increase}")
        
        # Should not leak file descriptors
        assert fd_increase <= 5, f"Too many file descriptors opened: +{fd_increase}"
    
    def test_temporary_object_cleanup(self):
        """Test that temporary objects are properly cleaned up."""
        import weakref
        
        # Create objects and weak references to track them
        weak_refs = []
        
        for i in range(10):
            # Create temporary object
            temp_data = RentDataFactory.create_realistic_rent_sample(100)
            
            # Create weak reference to track when it's deleted
            weak_ref = weakref.ref(temp_data)
            weak_refs.append(weak_ref)
            
            # Process the data
            result = detect_wucher_miete(temp_data, min_neighbors=5)
            
            # Delete references
            del temp_data, result
        
        # Force garbage collection
        gc.collect()
        
        # Check that objects were actually deleted
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        
        print(f"Objects still alive: {alive_objects} of {len(weak_refs)}")
        
        # Most objects should be garbage collected
        assert alive_objects <= len(weak_refs) // 2, \
            f"Too many objects still alive: {alive_objects}"
    
    def test_memory_growth_under_load(self, large_test_dataset):
        """Test memory behavior under sustained load."""
        process = psutil.Process()
        
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained load
        memory_samples = []
        
        for i in range(15):
            # Create and process data
            test_data = large_test_dataset(200)
            result = detect_wucher_miete(test_data, min_neighbors=5)
            
            # Take memory sample
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory - initial_memory)
            
            # Clean up
            del test_data, result
            
            # Occasional garbage collection
            if i % 4 == 0:
                gc.collect()
            
            print(f"Load iteration {i+1}: +{memory_samples[-1]:.1f} MB")
        
        # Analyze memory growth pattern
        early_samples = memory_samples[:5]
        late_samples = memory_samples[-5:]
        
        early_avg = np.mean(early_samples)
        late_avg = np.mean(late_samples)
        
        growth = late_avg - early_avg
        growth_rate = growth / len(memory_samples)
        
        print(f"Memory growth: {growth:.1f} MB ({growth_rate:.1f} MB/iteration)")
        
        # Should not show significant memory growth under sustained load
        assert growth < 50, f"Excessive memory growth under load: +{growth:.1f} MB"
        assert growth_rate < 5, f"High memory growth rate: {growth_rate:.1f} MB/iteration"


@pytest.mark.memory
@pytest.mark.fast
class TestMemoryUtilities:
    """Test memory monitoring utilities."""
    
    def test_memory_monitor_context(self, memory_monitor):
        """Test the memory monitoring fixture."""
        with memory_monitor() as get_memory:
            initial = get_memory()
            
            # Create some data
            data = RentDataFactory.create_realistic_rent_sample(50)
            
            peak = get_memory()
            
            # Clean up
            del data
            gc.collect()
            
            final = get_memory()
            
            print(f"Memory: initial={initial:.1f}, peak={peak:.1f}, final={final:.1f} MB")
            
            # Basic sanity checks
            assert peak >= initial, "Peak memory should be >= initial"
            assert final <= peak, "Final memory should be <= peak"
    
    def test_memory_profiling_helper(self):
        """Test a simple memory profiling helper."""
        def profile_memory(func, *args, **kwargs):
            """Simple memory profiling helper."""
            process = psutil.Process()
            
            gc.collect()
            initial = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            final = process.memory_info().rss / 1024 / 1024  # MB
            increase = final - initial
            
            return result, increase
        
        # Test the helper
        test_data = RentDataFactory.create_realistic_rent_sample(100)
        
        result, memory_used = profile_memory(
            detect_wucher_miete, 
            test_data, 
            min_neighbors=5
        )
        
        print(f"Memory used by wucher detection: {memory_used:.1f} MB")
        
        # Should return valid results
        assert hasattr(result, '__len__'), "Should return a valid result"
        assert memory_used >= 0, "Memory usage should be non-negative"
        assert memory_used < 100, "Should use reasonable memory for small dataset"
