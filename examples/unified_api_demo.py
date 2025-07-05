#!/usr/bin/env python3
"""
Demonstration of the unified BlowChoc API

This script shows how to use the unified API that can switch between
the original numba implementation and the new Rust implementation.
"""

import numpy as np
import time
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

from ferric_bloom.unified_filter import BlowChocFilter

def demo_basic_usage():
    """Demonstrate basic usage of the unified API."""
    print("=== Basic Usage Demo ===")
    
    # Create a filter with default numba backend
    print("Creating BlowChoc filter with numba backend...")
    filter_obj = BlowChocFilter(
        universe=4**21,  # For 21-mers
        nsubfilters=1,
        nblocks=1000,
        nchoices=2,
        nbits=8,
        backend='numba'
    )
    
    print(f"Created filter: {filter_obj}")
    print(f"Backend: {filter_obj.backend}")
    print(f"Memory usage: {filter_obj.mem_bytes} bytes")
    print()
    
    # Insert some test data
    test_keys = [12345, 67890, 11111, 22222, 33333]
    print("Inserting test keys...")
    for key in test_keys:
        filter_obj.insert(key)
    
    # Test lookups
    print("Testing lookups...")
    for key in test_keys:
        result = filter_obj.lookup(key)
        print(f"  Key {key}: {'Found' if result else 'Not found'}")
    
    # Test a key that wasn't inserted
    test_key = 99999
    result = filter_obj.lookup(test_key)
    print(f"  Key {test_key} (not inserted): {'Found' if result else 'Not found'}")
    
    print(f"Fill rate: {filter_obj.get_fill():.4f}")
    print(f"False positive rate: {filter_obj.get_fpr():.6f}")
    print()

def demo_backend_switching():
    """Demonstrate switching between backends."""
    print("=== Backend Switching Demo ===")
    
    # Create filter with numba backend
    filter_obj = BlowChocFilter(
        universe=4**15,  # Smaller universe for demo
        nsubfilters=1,
        nblocks=100,
        nchoices=2,
        nbits=8,
        backend='numba'
    )
    
    print(f"Initial backend: {filter_obj.backend}")
    
    # Insert some data
    test_keys = [1000, 2000, 3000, 4000, 5000]
    print("Inserting test keys with numba backend...")
    for key in test_keys:
        filter_obj.insert(key)
    
    print(f"Numba backend stats:\n{filter_obj.stats()}")
    print()
    
    # Switch to Rust backend
    try:
        print("Switching to Rust backend...")
        filter_obj.use_backend('rust')
        print(f"New backend: {filter_obj.backend}")
        
        # Re-insert the same data (filter is recreated)
        print("Re-inserting test keys with Rust backend...")
        for key in test_keys:
            filter_obj.insert(key)
        
        print(f"Rust backend stats:\n{filter_obj.stats()}")
        print()
        
        # Test lookups with Rust backend
        print("Testing lookups with Rust backend...")
        for key in test_keys:
            result = filter_obj.lookup(key)
            print(f"  Key {key}: {'Found' if result else 'Not found'}")
        
    except ImportError as e:
        print(f"Rust backend not available: {e}")
    print()

def demo_performance_comparison():
    """Compare performance between backends."""
    print("=== Performance Comparison Demo ===")
    
    # Parameters for the test
    universe = 4**20
    nblocks = 1000
    nchoices = 2
    nbits = 8
    num_operations = 10000
    
    # Generate test data
    np.random.seed(42)
    test_keys = np.random.randint(0, universe, num_operations, dtype=np.uint64)
    
    # Test numba backend
    print("Testing numba backend...")
    filter_numba = BlowChocFilter(
        universe=universe,
        nsubfilters=1,
        nblocks=nblocks,
        nchoices=nchoices,
        nbits=nbits,
        backend='numba'
    )
    
    start_time = time.time()
    for key in test_keys:
        filter_numba.insert(key)
    numba_insert_time = time.time() - start_time
    
    start_time = time.time()
    for key in test_keys:
        filter_numba.lookup(key)
    numba_lookup_time = time.time() - start_time
    
    print(f"Numba - Insert: {numba_insert_time:.4f}s ({num_operations/numba_insert_time:.0f} ops/s)")
    print(f"Numba - Lookup: {numba_lookup_time:.4f}s ({num_operations/numba_lookup_time:.0f} ops/s)")
    print(f"Numba - Fill rate: {filter_numba.get_fill():.4f}")
    print(f"Numba - FPR: {filter_numba.get_fpr():.6f}")
    print()
    
    # Test Rust backend
    try:
        print("Testing Rust backend...")
        filter_rust = BlowChocFilter(
            universe=universe,
            nsubfilters=1,
            nblocks=nblocks,
            nchoices=nchoices,
            nbits=nbits,
            backend='rust'
        )
        
        start_time = time.time()
        for key in test_keys:
            filter_rust.insert(key)
        rust_insert_time = time.time() - start_time
        
        start_time = time.time()
        for key in test_keys:
            filter_rust.lookup(key)
        rust_lookup_time = time.time() - start_time
        
        print(f"Rust - Insert: {rust_insert_time:.4f}s ({num_operations/rust_insert_time:.0f} ops/s)")
        print(f"Rust - Lookup: {rust_lookup_time:.4f}s ({num_operations/rust_lookup_time:.0f} ops/s)")
        print(f"Rust - Fill rate: {filter_rust.get_fill():.4f}")
        print(f"Rust - FPR: {filter_rust.get_fpr():.6f}")
        print()
        
        # Compare performance
        print("Performance comparison:")
        print(f"Rust insert speedup: {numba_insert_time/rust_insert_time:.2f}x")
        print(f"Rust lookup speedup: {numba_lookup_time/rust_lookup_time:.2f}x")
        
    except ImportError as e:
        print(f"Rust backend not available: {e}")
    print()

def demo_cost_functions():
    """Demonstrate different cost functions."""
    print("=== Cost Function Demo ===")
    
    # Test different cost function parameters
    configs = [
        {'sigma': 1.0, 'theta': 0.5, 'beta': -1.0, 'mu': -1.0, 'name': 'MinFPR'},
        {'sigma': -1.0, 'theta': -1.0, 'beta': 1.0, 'mu': 2.0, 'name': 'LookAhead'},
        {'sigma': -1.0, 'theta': -1.0, 'beta': -1.0, 'mu': -1.0, 'name': 'MinLoad'},
    ]
    
    universe = 4**15
    nblocks = 100
    nchoices = 3
    nbits = 6
    
    for config in configs:
        print(f"Testing {config['name']} cost function...")
        
        filter_obj = BlowChocFilter(
            universe=universe,
            nsubfilters=1,
            nblocks=nblocks,
            nchoices=nchoices,
            nbits=nbits,
            sigma=config['sigma'],
            theta=config['theta'],
            beta=config['beta'],
            mu=config['mu'],
            backend='numba'
        )
        
        # Insert test data
        test_keys = [i * 1000 for i in range(50)]
        for key in test_keys:
            filter_obj.insert(key)
        
        print(f"  Fill rate: {filter_obj.get_fill():.4f}")
        print(f"  FPR: {filter_obj.get_fpr():.6f}")
        print()

def main():
    """Run all demonstrations."""
    print("BlowChoc Unified API Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_backend_switching()
    demo_performance_comparison()
    demo_cost_functions()
    
    print("Demo complete!")

if __name__ == "__main__":
    main() 