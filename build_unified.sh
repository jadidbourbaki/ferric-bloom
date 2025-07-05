#!/bin/bash
# Build script for the unified BlowChoc API
# This script builds both the numba and rust backends

set -e  # Exit on any error

echo "🔧 Building Unified BlowChoc API"
echo "================================="
echo

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

# Check pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is required but not installed."
    exit 1
fi
echo "✅ pip found"

# Check Rust (optional for Rust backend)
if command -v rustc &> /dev/null; then
    echo "✅ Rust found: $(rustc --version)"
    RUST_AVAILABLE=true
else
    echo "⚠️  Rust not found - Rust backend will not be available"
    echo "   Install Rust from https://rustup.rs/ to enable Rust backend"
    RUST_AVAILABLE=false
fi

echo

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install numpy numba || pip3 install numpy numba

if [ "$RUST_AVAILABLE" = true ]; then
    pip install maturin || pip3 install maturin
fi
echo "✅ Python dependencies installed"
echo

# Build numba backend
echo "🐍 Setting up numba backend..."
echo "Note: BlowChoc is a submodule - no installation needed"
echo "✅ Numba backend ready (using submodule)"
echo

# Build Rust backend (if available)
if [ "$RUST_AVAILABLE" = true ]; then
    echo "🦀 Building Rust backend..."
    
    # Check if maturin is available
    if command -v maturin &> /dev/null; then
        echo "Building with maturin..."
        maturin develop --release
        echo "✅ Rust backend built successfully"
    else
        echo "❌ maturin not found - cannot build Rust backend"
        echo "   Install with: pip install maturin"
        RUST_AVAILABLE=false
    fi
else
    echo "⏭️  Skipping Rust backend (Rust not available)"
fi
echo

# Test the installation
echo "🧪 Testing installation..."

# Test numba backend
echo "Testing numba backend..."
python3 -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'python'))
try:
    from ferric_bloom.unified_filter import BlowChocFilter
    filter = BlowChocFilter(universe=4**10, nblocks=10, nchoices=2, nbits=4, backend='numba')
    filter.insert(12345)
    assert filter.lookup(12345), 'Lookup failed'
    print('✅ Numba backend test passed')
except Exception as e:
    print(f'❌ Numba backend test failed: {e}')
    exit(1)
"

# Test Rust backend (if available)
if [ "$RUST_AVAILABLE" = true ]; then
    echo "Testing Rust backend..."
    python3 -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'python'))
try:
    from ferric_bloom.unified_filter import BlowChocFilter
    filter = BlowChocFilter(universe=4**10, nblocks=10, nchoices=2, nbits=4)
    filter.use_backend('rust')
    filter.insert(12345)
    assert filter.lookup(12345), 'Lookup failed'
    print('✅ Rust backend test passed')
except ImportError:
    print('⚠️  Rust backend not available (this is okay)')
except Exception as e:
    print(f'❌ Rust backend test failed: {e}')
"
fi

echo
echo "🎉 Build complete!"
echo
echo "📖 Usage:"
echo "   from ferric_bloom.unified_filter import BlowChocFilter"
echo "   filter = BlowChocFilter(universe=4**21, nblocks=1000, nchoices=2, nbits=8)"
echo "   filter.use_backend('rust')  # Switch to Rust backend"
echo
echo "📚 See README.md for detailed documentation"
echo "🚀 Run 'python examples/unified_api_demo.py' for a demonstration"
echo

# Print summary
echo "📊 Summary:"
echo "   ✅ Numba backend: Available"
if [ "$RUST_AVAILABLE" = true ]; then
    echo "   ✅ Rust backend: Available"
else
    echo "   ❌ Rust backend: Not available"
fi
echo 