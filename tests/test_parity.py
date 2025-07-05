#!/usr/bin/env python3
"""
Parity test for ferric-bloom unified API.

This test ensures that both the Rust and Numba backends produce identical results
for the same inputs, verifying API consistency and correctness.
"""

import sys
import os
import pytest
import numpy as np
from typing import List, Tuple, Any

# Add the python directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from ferric_bloom.unified_filter import BlowChocFilter
except ImportError as e:
    print(f"❌ Cannot import unified API: {e}")
    print("Run: ./build_unified.sh")
    sys.exit(1)


class TestParity:
    """Test suite to ensure both backends produce identical results."""
    
    def setup_method(self, method=None):
        """Setup method run before each test (pytest compatible)."""
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Internal method to set up test environment."""
        # Standard test configuration
        self.universe = 4**12  # 16,777,216
        self.nblocks = 100
        self.nchoices = 2
        self.nbits = 8
        
        # Create filters for both backends
        self.filter_rust = BlowChocFilter(
            universe=self.universe, 
            nblocks=self.nblocks, 
            nchoices=self.nchoices, 
            nbits=self.nbits
        )
        
        self.filter_numba = BlowChocFilter(
            universe=self.universe, 
            nblocks=self.nblocks, 
            nchoices=self.nchoices, 
            nbits=self.nbits
        )
        
        # Check if both backends are available
        available_backends = self.filter_rust.available_backends()
        
        if len(available_backends) < 2:
            if 'pytest' in sys.modules:
                pytest.skip(f"Need both backends for parity test. Available: {available_backends}")
            else:
                raise RuntimeError(f"Need both backends for parity test. Available: {available_backends}")
        
        # Set backends
        self.filter_rust.use_backend('rust')
        self.filter_numba.use_backend('numba')
        
        # Test data - deterministic to avoid randomness issues
        self.test_keys = [
            42, 1337, 9999, 123456, 987654,
            0, 1, 2**32-1, 2**63-1,  # Edge cases
            *range(1000, 1100),  # Sequential range
            *[i * 997 for i in range(50)]  # Sparse distribution
        ]
        
        # Keys that won't be inserted (for negative tests)
        self.absent_keys = [
            777777, 888888, 999999, 2**32, 2**40,
            *range(5000, 5050),  # Different range
            *[i * 991 for i in range(25)]  # Different sparse pattern
        ]
    
    def test_backend_availability(self):
        """Test that both backends are available and correctly set."""
        available = self.filter_rust.available_backends()
        assert 'rust' in available, "Rust backend not available"
        assert 'numba' in available, "Numba backend not available"
        
        assert self.filter_rust.current_backend() == 'rust'
        assert self.filter_numba.current_backend() == 'numba'
    
    def test_initial_state_parity(self):
        """Test that both filters start in identical states."""
        # Both should be empty
        assert self.filter_rust.count() == self.filter_numba.count() == 0
        
        # Both should have same configuration
        assert self.filter_rust.universe == self.filter_numba.universe
        assert self.filter_rust.nblocks == self.filter_numba.nblocks
        assert self.filter_rust.nchoices == self.filter_numba.nchoices
        assert self.filter_rust.nbits == self.filter_numba.nbits
        
        # Both should have same basic properties
        assert self.filter_rust.filtertype == self.filter_numba.filtertype
        assert self.filter_rust.filterbits == self.filter_numba.filterbits
        assert self.filter_rust.mem_bytes == self.filter_numba.mem_bytes
        
        # Initial FPR should be identical (likely 0)
        assert self.filter_rust.get_fpr() == self.filter_numba.get_fpr()
        assert self.filter_rust.get_fill() == self.filter_numba.get_fill()
    
    def test_single_insertion_parity(self):
        """Test that single insertions produce identical results."""
        test_key = 42
        
        # Insert in both
        self.filter_rust.insert(test_key)
        self.filter_numba.insert(test_key)
        
        # Both should find the key
        assert self.filter_rust.lookup(test_key) == self.filter_numba.lookup(test_key) == True
        
        # Both should have same count
        assert self.filter_rust.count() == self.filter_numba.count() == 1
        
        # Both should have same fill/FPR (within small tolerance for float comparison)
        assert abs(self.filter_rust.get_fill() - self.filter_numba.get_fill()) < 1e-10
        assert abs(self.filter_rust.get_fpr() - self.filter_numba.get_fpr()) < 1e-10
    
    def test_multiple_insertion_parity(self):
        """Test that multiple insertions produce identical results."""
        # Insert all test keys
        for key in self.test_keys:
            self.filter_rust.insert(key)
            self.filter_numba.insert(key)
        
        # Both should have same count
        assert self.filter_rust.count() == self.filter_numba.count() == len(self.test_keys)
        
        # Both should find all inserted keys
        for key in self.test_keys:
            rust_result = self.filter_rust.lookup(key)
            numba_result = self.filter_numba.lookup(key)
            assert rust_result == numba_result == True, f"Key {key} lookup mismatch"
        
        # Both should have same fill/FPR
        assert abs(self.filter_rust.get_fill() - self.filter_numba.get_fill()) < 1e-10
        assert abs(self.filter_rust.get_fpr() - self.filter_numba.get_fpr()) < 1e-10
    
    def test_lookup_parity_negative(self):
        """Test that lookups for non-inserted keys produce identical results."""
        # Insert some keys
        for key in self.test_keys[:50]:  # Insert only half
            self.filter_rust.insert(key)
            self.filter_numba.insert(key)
        
        # Test lookups for absent keys
        for key in self.absent_keys:
            rust_result = self.filter_rust.lookup(key)
            numba_result = self.filter_numba.lookup(key)
            assert rust_result == numba_result, f"Key {key} absent lookup mismatch: rust={rust_result}, numba={numba_result}"
    
    def test_histogram_parity(self):
        """Test that fill histograms are identical."""
        # Insert test data
        for key in self.test_keys:
            self.filter_rust.insert(key)
            self.filter_numba.insert(key)
        
        # Get histograms
        rust_hist = self.filter_rust.get_fill_histogram()
        numba_hist = self.filter_numba.get_fill_histogram()
        
        # Convert to numpy arrays for comparison
        rust_hist = np.array(rust_hist)
        numba_hist = np.array(numba_hist)
        
        # Should be identical
        assert np.array_equal(rust_hist, numba_hist), "Fill histograms don't match"
    
    def test_array_parity(self):
        """Test that underlying arrays are identical."""
        # Insert test data
        for key in self.test_keys:
            self.filter_rust.insert(key)
            self.filter_numba.insert(key)
        
        # Get raw arrays
        rust_array = np.array(self.filter_rust.array)
        numba_array = np.array(self.filter_numba.array)
        
        # Should be identical
        assert np.array_equal(rust_array, numba_array), "Underlying arrays don't match"
    
    def test_edge_cases_parity(self):
        """Test edge cases for parity."""
        edge_cases = [
            0,  # Zero
            1,  # One
            2**32 - 1,  # 32-bit max
            2**63 - 1,  # 63-bit max (avoid 64-bit signed overflow)
        ]
        
        for key in edge_cases:
            # Fresh filters for each test - universe must be power of 4
            f_rust = BlowChocFilter(universe=4**8, nblocks=10, nchoices=2, nbits=4)  # 4^8 = 65536
            f_numba = BlowChocFilter(universe=4**8, nblocks=10, nchoices=2, nbits=4)
            f_rust.use_backend('rust')
            f_numba.use_backend('numba')
            
            # Insert and test
            f_rust.insert(key)
            f_numba.insert(key)
            
            assert f_rust.lookup(key) == f_numba.lookup(key) == True, f"Edge case {key} failed"
            assert f_rust.count() == f_numba.count() == 1, f"Edge case {key} count mismatch"
    
    def test_parameter_configurations_parity(self):
        """Test different parameter configurations for parity."""
        configurations = [
            (4**5, 10, 1, 4),    # Small, no choices (4^5 = 1024)
            (4**5, 10, 2, 4),    # Small, 2 choices (4^5 = 1024)
            (4**7, 50, 3, 8),    # Medium, 3 choices (4^7 = 16384)
            (4**8, 100, 2, 12),  # Large, 2 choices (4^8 = 65536)
        ]
        
        for universe, nblocks, nchoices, nbits in configurations:
            f_rust = BlowChocFilter(universe=universe, nblocks=nblocks, nchoices=nchoices, nbits=nbits)
            f_numba = BlowChocFilter(universe=universe, nblocks=nblocks, nchoices=nchoices, nbits=nbits)
            
            # Skip if backends not available
            if len(f_rust.available_backends()) < 2:
                continue
                
            f_rust.use_backend('rust')
            f_numba.use_backend('numba')
            
            # Test with same data
            test_data = range(0, 100, 3)  # Sparse data
            
            for key in test_data:
                f_rust.insert(key)
                f_numba.insert(key)
            
            # Verify parity
            assert f_rust.count() == f_numba.count(), f"Config {configurations} count mismatch"
            assert abs(f_rust.get_fill() - f_numba.get_fill()) < 1e-10, f"Config {configurations} fill mismatch"
            assert abs(f_rust.get_fpr() - f_numba.get_fpr()) < 1e-10, f"Config {configurations} FPR mismatch"
            
            # Test lookups
            for key in test_data:
                assert f_rust.lookup(key) == f_numba.lookup(key) == True, f"Config {configurations} lookup mismatch for {key}"
    
    def test_sequential_operations_parity(self):
        """Test that sequential operations maintain parity."""
        keys_to_insert = list(range(100))
        keys_to_check = list(range(150))  # Includes some not inserted
        
        for i, key in enumerate(keys_to_insert):
            # Insert in both
            self.filter_rust.insert(key)
            self.filter_numba.insert(key)
            
            # Check counts match
            assert self.filter_rust.count() == self.filter_numba.count() == i + 1
            
            # Check that all previously inserted keys are still found
            for prev_key in keys_to_insert[:i+1]:
                assert self.filter_rust.lookup(prev_key) == self.filter_numba.lookup(prev_key) == True
        
        # Final comprehensive check
        for key in keys_to_check:
            rust_result = self.filter_rust.lookup(key)
            numba_result = self.filter_numba.lookup(key)
            assert rust_result == numba_result, f"Sequential test key {key} mismatch"


def run_parity_tests():
    """Run all parity tests and report results."""
    print("🔍 Running API Parity Tests")
    print("=" * 50)
    
    # Check if unified API is available
    try:
        test_filter = BlowChocFilter(universe=4**5, nblocks=10, nchoices=2, nbits=4)  # 4^5 = 1024
        available_backends = test_filter.available_backends()
        print(f"📊 Available backends: {available_backends}")
        
        if len(available_backends) < 2:
            print("❌ Need both backends for parity testing")
            print("   Available backends:", available_backends)
            if 'rust' not in available_backends:
                print("   Missing: Rust backend - run: maturin develop --release")
            if 'numba' not in available_backends:
                print("   Missing: Numba backend - run: cd blowchoc && pip install -e .")
            print("   Or run: ./build_unified.sh")
            return False
            
    except Exception as e:
        print(f"❌ Cannot initialize unified API: {e}")
        return False
    
    # Run tests
    test_cases = [
        "test_backend_availability",
        "test_initial_state_parity", 
        "test_single_insertion_parity",
        "test_multiple_insertion_parity",
        "test_lookup_parity_negative",
        "test_histogram_parity",
        "test_array_parity",
        "test_edge_cases_parity",
        "test_parameter_configurations_parity",
        "test_sequential_operations_parity"
    ]
    
    test_instance = TestParity()
    
    passed = 0
    failed = 0
    
    for test_name in test_cases:
        try:
            print(f"🧪 {test_name}...", end=" ")
            # Setup fresh environment for each test
            test_instance._setup_test_environment()
            test_method = getattr(test_instance, test_name)
            test_method()
            print("✅ PASS")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All parity tests passed! Both backends behave identically.")
        return True
    else:
        print("⚠️  Some parity tests failed. Backend behavior differs.")
        return False


if __name__ == "__main__":
    success = run_parity_tests()
    sys.exit(0 if success else 1) 