---
layout: home
title: Ferric Bloom
---

# Ferric Bloom

[![Sponsor](https://img.shields.io/badge/Sponsor-❤️-pink.svg)](https://github.com/sponsors/jadidbourbaki)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Python Bindings](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

A **high-performance Bloom filter library** written in Rust with cache-optimized blocked designs. Implements [Blocked Bloom filters with Choices](https://arxiv.org/abs/2501.18977), or simply blowchoc, a data structure invented by the [Rahmann Lab](https://www.rahmannlab.de/index.html) at Saarland University. This is designed to be an optimized version of blowchoc's original Python + Numba implementation. Ferric Bloom delivers **34,000× faster insertions** and **3.8× faster queries** compared to blowchoc's original Python + Numba implementation, with potentially more performance gains once we add SIMD and parallelization.

## Features

- **Blocked Bloom Filter**: 512-bit cache-aligned blocks for optimal performance
- **Standard Bloom Filter**: Classic implementation with modern optimizations  
- **Python Bindings**: Easy integration via PyO3
- **Zero Runtime Compilation**: Native code, no JIT overhead
- **Multiple Cost Functions**: Intelligent block selection strategies

## Performance Results

Comprehensive benchmarks against **blowchoc** (state-of-the-art Python+Numba implementation):

### Insert Performance
<img src="tests/ferric_vs_blowchoc_insert_throughput.png" alt="Insert Throughput" width="400">
<img src="tests/ferric_vs_blowchoc_insert_time.png" alt="Insert Time" width="400">

### Query Performance  
<img src="tests/ferric_vs_blowchoc_query_throughput.png" alt="Query Throughput" width="400">
<img src="tests/ferric_vs_blowchoc_query_time.png" alt="Query Time" width="400">

### Key Results (200 elements)
- **Insert Performance**: **34,860× faster** (9.9M vs 285 ops/sec)
- **Query Performance**: **3.8× faster** (11.6M vs 3.1M ops/sec)  
- **Insert Time**: 0.020ms vs 702ms (**35,000× improvement**)

## Quick Start

### Installation

```bash
# Rust
cargo add ferric-bloom

# Python  
pip install maturin
maturin develop --features python
```

### Usage

```rust
// Rust - Blocked Bloom Filter
use ferric_bloom::BlockedBloomFilter;

let mut filter = BlockedBloomFilter::new(100, 8, 2)?; // 100 blocks, 8 bits/key, 2 choices
filter.insert(42);
assert!(filter.contains(42));
```

```python
# Python - Blocked Bloom Filter  
import ferric_bloom as fb

filter = fb.BlockedBloomFilter(100, 8, 2)
filter.insert(42)
print(filter.contains(42))  # True
```

## Why So Fast?

1. **Cache-Optimized**: 512-bit blocks aligned to CPU cache lines
2. **Zero JIT Overhead**: Native code compilation, no runtime warmup
3. **Memory Efficient**: Direct bit manipulation without Python boxing
4. **Smart Block Selection**: Multiple cost functions for load balancing

## Advanced Features

```rust
use ferric_bloom::{BlockedBloomFilter, blocked_bloom::CostFunction};

let mut filter = BlockedBloomFilter::new(100, 8, 2)?;

// Set cost function for intelligent block selection
filter.set_cost_function(CostFunction::MinFpr { sigma: 1.0, theta: 1.0 });
// Or: CostFunction::MinLoad
// Or: CostFunction::LookAhead { mu: 1.5 }

// Get detailed statistics
println!("{}", filter.stats());
```

## Testing & Benchmarks

```bash
# Run Rust tests
cargo test

# Run benchmarks  
cargo bench

# Compare with blowchoc (requires setup)
cd tests && python ferric_vs_blowchoc_benchmark.py
```

*Detailed setup instructions: [tests/README.md](tests/README.md)*

## Acknowledgments

Inspired by the [blowchoc](https://github.com/BloomfilterTeam/blowchoc-filters) project and cache-efficient design principles.

## Troubleshooting

1. **Always run tests from `tests/` directory** to avoid import path conflicts
2. **Use editable mode for blowchoc** (`pip install -e .`) not regular mode
3. **Install all dependencies** before installing packages
4. **Use Python 3.12+** for best compatibility with both packages 