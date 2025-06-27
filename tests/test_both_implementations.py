import ferric_bloom as fb

# Try importing blowchoc (may fail due to missing dependencies)
try:
    from blowchoc.filter_bb_choices import build_filter as build_blocked_filter
    import numpy as np
    blowchoc_available = True
    print("✅ Both ferric-bloom and blowchoc available")
except ImportError as e:
    blowchoc_available = False
    print(f"⚠️  Only ferric-bloom available: {e}")

# Test ferric-bloom (always works)
print("\n=== Ferric-bloom BlockedBloomFilter ===")
ferric_filter = fb.BlockedBloomFilter(100, 8, 2)
ferric_filter.insert(42)
result = ferric_filter.contains(42)
print(f"Parameters: 100 blocks, 8 bits per key, 2 choices")
print(f"Architecture: 512-bit cache-aligned blocks, Rust implementation")
print(f"Test (insert 42, query 42): {result}")
print(f"Stats: {ferric_filter.stats()}")

# Test blowchoc (if available)
if blowchoc_available:
    print("\n=== Blowchoc BBChoicesFilter (Blocked Bloom with Choices) ===")
    # Create blowchoc blocked bloom filter with choices (BBChoicesFilter)
    # This is the proper comparison to ferric-bloom's BlockedBloomFilter
    universe = 4**20  # Large universe for general keys
    nsubfilters = 5   # Number of parallel subfilters  
    nblocks = 100     # Number of blocks (same as ferric-bloom)
    nchoices = 2      # Number of choices (same as ferric-bloom)
    nbits = 8         # Bits per element (same as ferric-bloom)
    
    # Build blocked bloom filter with choices
    blowchoc_filter = build_blocked_filter(
        universe=universe,
        nsubfilters=nsubfilters, 
        nblocks=nblocks,
        nchoices=nchoices,
        nbits=nbits,
        sigma=1.0,  # Cost function parameter
        theta=1.0   # Cost function parameter
    )
    bit_positions = np.empty(nbits, dtype=np.uint64)
    
    # Insert and test
    blowchoc_filter.insert(blowchoc_filter.array, 42, bit_positions)
    result = blowchoc_filter.lookup(blowchoc_filter.array, 42, bit_positions)
    print(f"Parameters: {nblocks} blocks, {nbits} bits per key, {nchoices} choices")
    print(f"Architecture: 512-bit blocks, Python+Numba implementation")
    print(f"Cost function: sigma={1.0}, theta={1.0}")
    print(f"Test (insert 42, query 42): {bool(result)}")
    print(f"Total bits: {blowchoc_filter.filterbits}, Memory: {blowchoc_filter.mem_bytes} bytes")
    
    print("\n=== Comparison Summary ===")
    print("Both implementations use:")
    print("- 512-bit block size (cache-aligned)")
    print("- Multiple choices for block selection")
    print("- Similar API: insert(key), contains/lookup(key)")
    print("- Cost functions for intelligent block selection")
    print("\nKey differences:")
    print("- ferric-bloom: Pure Rust, compile-time optimization")
    print("- blowchoc: Python+Numba, runtime JIT compilation")
else:
    print("Skipping blowchoc tests")
