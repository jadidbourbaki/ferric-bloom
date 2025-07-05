# Ferric Bloom

[![Sponsor](https://img.shields.io/badge/Sponsor-❤️-pink.svg)](https://github.com/sponsors/jadidbourbaki)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Python Bindings](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

A **high-performance Bloom filter library** written in Rust with cache-optimized blocked designs. Implements [Blocked Bloom filters with Choices](https://arxiv.org/abs/2501.18977), or simply blowchoc, a data structure invented by the [Rahmann Lab](https://www.rahmannlab.de/index.html) at Saarland University. This is designed to be an optimized version of blowchoc's original Python + Numba implementation. Ferric Bloom delivers **bit-level identical results** to the original BlowChoc implementation while providing **1.5-2.2× performance improvements** across all operations.

**Features a unified Python API** that allows seamless switching between the original BlowChoc numba backend and the new high-performance Rust backend, providing the best of both worlds: mature, battle-tested algorithms with cutting-edge performance.

## Features

- **Perfect Algorithmic Parity**: Bit-level identical results to original BlowChoc
- **Blocked Bloom Filter**: 512-bit cache-aligned blocks for optimal performance
- **Unified Python API**: Switch between numba and Rust backends seamlessly
- **Backend Compatibility**: Full API compatibility with original BlowChoc
- **Zero Runtime Compilation**: Native code, no JIT overhead
- **Multiple Cost Functions**: Intelligent block selection strategies


## Performance Results

Comprehensive benchmarks against **blowchoc** (state-of-the-art Python+Numba implementation). We attempt
to faithfully replicate the methodology of Section 4.2 from the original Rahmann lab BlowChoc filter paper.

<img src="tests/paper_faithful_performance_comparison.png" alt="Section 4.2 from paper">

## Quick Start

### Installation

#### Rust Library
```bash
cargo add ferric-bloom
```

#### Python API
```bash
# Install dependencies
pip install numpy numba maturin

# Build Rust backend (optional - for maximum performance)
maturin develop --release

# Use the automated setup script
./build_unified.sh
```

The unified API works with or without the Rust backend. If Rust is not available, it falls back to the original numba implementation.

### Usage

#### Rust Library
```rust
// Rust - Blocked Bloom Filter
use ferric_bloom::BlockedBloomFilter;

let mut filter = BlockedBloomFilter::new(100, 8, 2)?; // 100 blocks, 8 bits/key, 2 choices
filter.insert(42);
assert!(filter.contains(42));
```

#### Python API
```python
# Import the unified API
from ferric_bloom.unified_filter import BlowChocFilter

# Create a filter with default numba backend
filter = BlowChocFilter(
    universe=4**21,    # Size of key universe (for 21-mers)
    nblocks=1000,      # Number of blocks
    nchoices=2,        # Number of choices
    nbits=8            # Bits per key
)

# Insert and query keys
filter.insert(42)
print(filter.lookup(42))  # True

# Switch to high-performance Rust backend
filter.use_backend('rust')
filter.insert(123)
print(filter.lookup(123))  # True

# Get statistics
print(f"Fill rate: {filter.get_fill():.4f}")
print(f"False positive rate: {filter.get_fpr():.6f}")
```

### Backend Switching
```python
from ferric_bloom.unified_filter import BlowChocFilter

# Create filter with numba backend
filter = BlowChocFilter(universe=4**20, nblocks=1000, nchoices=2, nbits=8)

# Insert data with numba backend
for i in range(1000):
    filter.insert(i)

print(f"Numba backend - Fill rate: {filter.get_fill():.4f}")

# Switch to Rust backend for maximum performance
filter.use_backend('rust')

# Re-insert data with Rust backend (filter is recreated)
for i in range(1000):
    filter.insert(i)

print(f"Rust backend - Fill rate: {filter.get_fill():.4f}")
```

### Performance Comparison
```python
import time
import numpy as np

# Test data
test_keys = np.random.randint(0, 2**32, 10000, dtype=np.uint64)

# Benchmark numba backend
filter_numba = BlowChocFilter(universe=4**20, nblocks=1000, nchoices=2, nbits=8, backend='numba')
start = time.time()
for key in test_keys:
    filter_numba.insert(key)
numba_time = time.time() - start

# Benchmark rust backend
filter_rust = BlowChocFilter(universe=4**20, nblocks=1000, nchoices=2, nbits=8, backend='rust')
start = time.time()
for key in test_keys:
    filter_rust.insert(key)
rust_time = time.time() - start

print(f"Rust speedup: {numba_time/rust_time:.1f}x")
```

### API Compatibility

The unified API maintains full compatibility with the original BlowChoc interface:

```python
# Original BlowChoc code works unchanged
from ferric_bloom.unified_filter import BlowChocFilter

filter = BlowChocFilter(universe=4**21, nsubfilters=1, nblocks=1000, 
                       nchoices=2, nbits=8, backend='numba')

# All original methods work
filter.insert(key)
result = filter.lookup(key)
fpr = filter.get_fpr()
fill = filter.get_fill()
histogram = filter.get_fill_histogram()

# Plus new convenience methods
filter.add(key)         # Alias for insert
filter.contains(key)    # Alias for lookup
filter.clear()          # Clear all bits
print(filter.stats())   # Detailed statistics
```

### Backend Comparison

| Feature | Numba Backend | Rust Backend |
|---------|---------------|--------------|
| **Performance** | Good (JIT compiled) | Excellent (native) |
| **Memory Usage** | Higher | Lower |
| **Startup Time** | Slow first run | Fast |
| **Dependencies** | Python, numba | Rust toolchain |
| **Compatibility** | Full BlowChoc API | Full BlowChoc API |

## Why So Fast?

1. **Cache-Optimized**: 512-bit blocks aligned to CPU cache lines
2. **Zero JIT Overhead**: Native code compilation, no runtime warmup
3. **Memory Efficient**: Direct bit manipulation without Python boxing
4. **Smart Block Selection**: Multiple cost functions for load balancing

## Testing & Benchmarks

```bash
# Run Rust tests
cargo test

# Run benchmarks  
cargo bench

# Test unified API
python examples/unified_api_demo.py

# Test perfect algorithmic parity
cd tests && python test_parity.py

# Run comprehensive paper-faithful benchmarks (~15 minutes)
cd tests && python paper_faithful_benchmark.py
```

*Detailed setup instructions: [tests/README.md](tests/README.md)*

## Documentation

- [Python Bindings Guide](PYTHON_BINDINGS.md)
- [Performance Comparison](COMPARISON.md)
- [Testing Environment Setup](tests/README.md)

## Acknowledgments

Inspired by the [blowchoc](https://github.com/BloomfilterTeam/blowchoc-filters) project and cache-efficient design principles.

## Troubleshooting

1. **Always run tests from `tests/` directory** to avoid import path conflicts
2. **Use editable mode for blowchoc** (`pip install -e .`) not regular mode
3. **Install all dependencies** before installing packages
4. **Use Python 3.12+** for best compatibility with both packages