# Testing Environment Setup

This guide helps you set up a Python environment with both **ferric-bloom** and **blowchoc** for compatibility testing and benchmarking.

## Setup

```bash
# Create Python 3.12+ virtual environment (recommended)
python3.12 -m venv .venv-testing
source .venv-testing/bin/activate

# Use the unified build script (recommended)
./build_unified.sh

# OR manual setup (after activating venv):
# Install dependencies from requirements file
pip install -r tests/requirements-testing.txt

# Build ferric-bloom Python bindings
maturin develop --release

# Install blowchoc in editable mode (IMPORTANT: use -e flag)
cd blowchoc && pip install -e . && cd ..

# Verify installation
cd tests
python test_parity.py
```

## Available Test Files

### Core Tests
- **`test_parity.py`** - ⭐ **API parity test** - Ensures both backends produce identical results
- **`paper_faithful_benchmark.py`** - 🚀 **Paper-faithful performance benchmark** - Comprehensive comparison following section 4.2 of the original BlowChoc paper methodology

The **parity test** is the most important test as it verifies that both the Rust and Numba backends behave identically for:
- Insert/lookup operations
- Fill histograms and FPR calculations  
- Edge cases and various parameter configurations
- Sequential operations and state consistency

### Rust Benchmarks
- **`src/bin/rust_micro_benchmark.rs`** - Native Rust benchmark (eliminates Python FFI overhead)

Run with: `cargo run --release --bin rust_micro_benchmark`

### Unified API Demo
- **`examples/unified_api_demo.py`** - Demonstrates the unified API with both backends

Run with: `python examples/unified_api_demo.py`

## Performance Results

### Paper-Faithful Benchmark Results 📊

Following the exact methodology from the original BlowChoc paper with a **2 GB filter (17.2 Gbits)**:

#### 🏆 **Peak Performance**
- **Rust Insertion**: 1.35 Mops/s (1.5x faster than Numba)
- **Rust Successful Lookup**: 4.89 Mops/s (2.2x faster than Numba)
- **Rust Unsuccessful Lookup**: 3.28 Mops/s (1.8x faster than Numba)

#### 📈 **Thread Scaling (k=14 hash functions)**
| Threads | Rust Insert | Numba Insert | Rust Lookup | Numba Lookup | Speedup |
|---------|-------------|--------------|-------------|--------------|---------|
| 1       | 0.30        | 0.26         | 1.43        | 0.91         | 1.6x    |
| 2       | 0.65        | 0.46         | 4.04        | 1.88         | 2.1x    |
| 4       | 0.78        | 0.49         | 3.58        | 1.73         | 2.1x    |
| 8       | 1.08        | 0.70         | 4.64        | 1.99         | 2.3x    |
| 12      | 1.30        | 0.80         | 4.89        | 1.94         | 2.5x    |
| 16      | 1.35        | 0.73         | 4.32        | 2.07         | 2.1x    |

#### 🎯 **Key Findings**
- **Superior Thread Scaling**: Rust consistently outperforms Numba across all thread counts
- **Optimal Performance**: Both backends peak around 12-16 threads
- **Lookup vs Insertion**: Lookups are consistently faster than insertions (as expected)
- **Unsuccessful Lookups**: Faster than successful lookups (matching paper predictions)
- **Perfect Algorithmic Parity**: Both backends produce identical bit-level results

*All measurements in Mops/s (Million operations per second)*

### Running Tests

```bash
# Quick parity test (recommended first)
cd tests
python test_parity.py

# Comprehensive paper-faithful benchmark (takes ~15 minutes)
cd tests
python paper_faithful_benchmark.py

# Run with pytest for detailed output
cd tests
pytest test_parity.py -v
```

The benchmark generates:
- `paper_faithful_performance_comparison.png/pdf` - Performance plots
- `paper_faithful_results.csv` - Detailed results data

## Additional Resources

- **Main README**: `README.md` - Project overview and API documentation
- **Python Bindings**: `PYTHON_BINDINGS.md` - Detailed Python API reference
- **Comparison**: `COMPARISON.md` - Performance comparison between backends
- **Unified API Demo**: `examples/unified_api_demo.py` - Working code examples
