# Testing Environment Setup

This guide helps you set up a Python environment with both **ferric-bloom** and **blowchoc** for compatibility testing and benchmarking.

## Quick Setup

```bash
# Create Python 3.12+ virtual environment
python3.12 -m venv .venv-testing
source .venv-testing/bin/activate

# Install dependencies from requirements file
pip install -r tests/requirements-testing.txt

# Build ferric-bloom Python bindings
maturin develop --release

# Install blowchoc in editable mode (IMPORTANT: use -e flag)
cd blowchoc && pip install -e . && cd ..

# Verify installation
cd tests
python test_both_implementations.py
```

## Requirements File

The `tests/requirements-testing.txt` file contains all necessary dependencies:

```bash
# Install all testing dependencies
pip install -r tests/requirements-testing.txt
```

**Key dependencies:**
- `numpy>=2.0.0` - Core arrays and math
- `scipy>=1.10.0` - Scientific computing  
- `pandas>=2.0.0` - Data analysis for benchmarks
- `matplotlib>=3.5.0` - Plotting benchmark results
- `numba>=0.58.0` - JIT compilation for blowchoc
- `pytest>=7.0.0` - Testing framework
- `tqdm>=4.60.0` - Progress bars for benchmarks

## Available Test Files

### Core Tests
- **`test_both_implementations.py`** - Basic compatibility test between ferric-bloom and blowchoc
- **`ferric_vs_blowchoc_benchmark.py`** - Comprehensive performance comparison

### Rust Benchmarks
- **`src/bin/rust_micro_benchmark.rs`** - Native Rust benchmark (eliminates Python FFI overhead)

Run with: `cargo run --release --bin rust_micro_benchmark`

## Benchmarking

### Quick Benchmark
```bash
cd tests
python ferric_vs_blowchoc_benchmark.py
```

This generates 5 comparison plots:
- `ferric_vs_blowchoc_insert_throughput.pdf` - Insert operations per second
- `ferric_vs_blowchoc_query_throughput.pdf` - Query operations per second  
- `ferric_vs_blowchoc_insert_time.pdf` - Insert time in milliseconds
- `ferric_vs_blowchoc_query_time.pdf` - Query time in milliseconds
- `ferric_vs_blowchoc_creation_time.pdf` - Filter creation time

### Benchmark Range
- **Element counts**: 0-200 in steps of 25
- **Configuration**: 100 blocks, 8 bits per key, 2 choices
- **Comparison**: ferric-bloom (native Rust) vs blowchoc (Python+Numba)

## Implementation Comparison

### ferric-bloom (Rust)
- **Architecture**: 512-bit cache-aligned blocks
- **Language**: Pure Rust with Python bindings via PyO3
- **Strengths**: Extremely fast insertions, consistent performance
- **Installation**: `maturin develop --release`

### blowchoc (Python+Numba)
- **Architecture**: 512-bit blocks with cost function optimization
- **Language**: Python with Numba JIT compilation
- **Strengths**: Research-grade blocked bloom filter with choices
- **Installation**: `pip install -e .` (editable mode required)

## Why Editable Installation for blowchoc?

BlowChoc requires **editable mode installation** (`pip install -e .`) because:

1. **Package Structure**: BlowChoc has nested subdirectories (`lowlevel/`, `io/`) that aren't properly included in standard `pip install .`
2. **Development Design**: The package was designed for conda environments with editable installs
3. **Import Dependencies**: The modules have complex internal dependencies that need the full directory structure

## Common Issues

### Python Version Requirements
- **ferric-bloom**: Python 3.8+
- **blowchoc**: Python 3.12+ (due to dependency requirements)

### blowchoc Installation Problems

**Problem**: `No module named 'blowchoc.lowlevel'`
**Solution**: 
```bash
# Uninstall broken installation
pip uninstall blowchoc -y

# Install required dependencies first
pip install -r tests/requirements-testing.txt

# Use editable mode installation
cd blowchoc && pip install -e .
```

**Problem**: Import works but tests hang
**Solution**: Run tests from the `tests/` directory to avoid local directory conflicts:
```bash
cd tests
python test_both_implementations.py
```

### Import Errors
```python
# Test ferric-bloom import
try:
    import ferric_bloom
    print("✅ ferric-bloom OK")
except ImportError as e:
    print(f"❌ ferric-bloom failed: {e}")
    print("Run: maturin develop --release")

# Test blowchoc import (using blocked bloom filter)
try:
    from blowchoc.filter_bb_choices import build_filter
    print("✅ blowchoc OK")
except ImportError as e:
    print(f"❌ blowchoc failed: {e}")
    print("Run: cd blowchoc && pip install -e .")
```

## Environment Verification

```bash
# Check if everything is working
python -c "
import ferric_bloom as fb
print(f'Ferric-bloom version: {getattr(fb, \"__version__\", \"unknown\")}')

try:
    from blowchoc.filter_bb_choices import build_filter
    print('Blowchoc: Available (BBChoicesFilter)')
except ImportError as e:
    print(f'Blowchoc: Not available - {e}')

# Test basic functionality
bloom = fb.BlockedBloomFilter(100, 8, 2)  # 100 blocks, 8 bits/key, 2 choices
bloom.insert(123)
print(f'Basic test: {bloom.contains(123)}')
"
```

## Performance Testing Results

Both implementations are now properly installed and can be compared:

```bash
# Run compatibility tests
cd tests
python test_both_implementations.py

# Expected output:
# ✅ Both ferric-bloom and blowchoc available
# Ferric-bloom test: True
# Blowchoc test: True
```

### Typical Performance Results (200 elements)
- **Insert performance**: ferric-bloom ~30,000x faster than blowchoc
- **Query performance**: ferric-bloom ~3x faster than blowchoc (native Rust)
- **Creation time**: ferric-bloom near-instantaneous vs blowchoc ~seconds

*Note: Query performance through Python bindings may show different results due to FFI overhead*

## Troubleshooting

1. **Always run tests from `tests/` directory** to avoid import path conflicts
2. **Use editable mode for blowchoc** (`pip install -e .`) not regular mode
3. **Install all dependencies** before installing packages using `requirements-testing.txt`
4. **Use Python 3.12+** for best compatibility with both packages
5. **Use `--release` flag** with maturin for optimal ferric-bloom performance
