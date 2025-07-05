"""
Unified BlowChoc Filter API

This module provides a unified interface for BlowChoc filters that can switch
between the original numba implementation and the Rust implementation.

Usage:
    # Create a filter with default numba backend
    filter = BlowChocFilter(universe=4**21, nsubfilters=1, nblocks=1000, 
                           nchoices=2, nbits=8)
    
    # Switch to Rust backend
    filter.use_backend('rust')
    
    # Use the filter with identical API
    filter.insert(key, bit_positions)
    result = filter.lookup(key, bit_positions)
"""

import numpy as np
from typing import Optional, Union, Literal, Any
import warnings
import sys
import os

# Add the blowchoc submodule to the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_blowchoc_path = os.path.join(_project_root, 'blowchoc')
if _blowchoc_path not in sys.path:
    sys.path.insert(0, _blowchoc_path)

# Import the original numba implementation from submodule
try:
    from blowchoc.filter_bb_choices import build_filter as build_filter_numba
    from blowchoc.filter_bb_choices import BBChoicesFilter
    NUMBA_AVAILABLE = True
except ImportError as e:
    NUMBA_AVAILABLE = False
    build_filter_numba = None
    BBChoicesFilter = None
    print(f"Warning: Could not import numba BlowChoc implementation: {e}")

# Try to import the Rust implementation
try:
    from ._ferric_bloom import BlowChocExactFilter as RustBlowChocExactFilter
    from ._ferric_bloom import BlowChocExactFilterWithMultipliers as RustBlowChocExactFilterWithMultipliers
    RUST_AVAILABLE = True
except ImportError:
    try:
        from .ferric_bloom import BlowChocExactFilter as RustBlowChocExactFilter
        from .ferric_bloom import BlowChocExactFilterWithMultipliers as RustBlowChocExactFilterWithMultipliers
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False
        RustBlowChocExactFilter = None
        RustBlowChocExactFilterWithMultipliers = None


class BlowChocFilter:
    """
    Unified BlowChoc filter that can switch between numba and rust backends.
    
    This class provides the same API as the original BBChoicesFilter namedtuple
    but allows switching between different implementations.
    """
    
    def __init__(self, 
                 universe: int,
                 nsubfilters: int = 1,
                 nblocks: int = 1000,
                 nchoices: int = 2,
                 nbits: int = 8,
                 bitversion: str = 'random',
                 sigma: float = -1.0,
                 theta: float = -1.0,
                 beta: float = -1.0,
                 mu: float = -1.0,
                 backend: Literal['numba', 'rust'] = 'numba',
                 **kwargs):
        """
        Initialize BlowChoc filter with given parameters.
        
        Args:
            universe: Size of key universe (e.g., 4**k for k-mers)
            nsubfilters: Number of subfilters for parallelization
            nblocks: Total number of blocks
            nchoices: Number of choices (alternative block locations)
            nbits: Number of bits set per key
            bitversion: Bit version strategy ('random', 'distinct', 'double')
            sigma: Weight parameter for cost function
            theta: Weight parameter for cost function  
            beta: Parameter for alternative cost function
            mu: Parameter for alternative cost function
            backend: Backend to use ('numba' or 'rust')
            **kwargs: Additional parameters passed to underlying implementation
        """
        self.universe = universe
        self.nsubfilters = nsubfilters
        self.nblocks = nblocks
        self.nchoices = nchoices
        self.nbits = nbits
        self.bitversion = bitversion
        self.sigma = sigma
        self.theta = theta
        self.beta = beta
        self.mu = mu
        self.kwargs = kwargs
        
        # Current backend and implementation
        self._backend = None
        self._filter = None
        self._bit_positions_cache = None
        self._insertion_count = 0  # Track insertions for numba backend
        
        # Set the backend
        self.use_backend(backend)
    
    def use_backend(self, backend: Literal['numba', 'rust']) -> None:
        """
        Switch to a different backend implementation.
        
        Args:
            backend: Backend to use ('numba' or 'rust')
        """
        if backend == 'rust' and not RUST_AVAILABLE:
            raise ImportError(
                "Rust backend not available. Build ferric-bloom with: "
                "maturin develop"
            )
        
        if backend == 'numba' and not NUMBA_AVAILABLE:
            raise ImportError(
                "Numba backend not available. Install dependencies: "
                "pip install numpy numba"
            )
        
        if backend == self._backend:
            return  # Already using this backend
            
        if backend == 'numba':
            self._setup_numba_backend()
            self._insertion_count = 0
        elif backend == 'rust':
            self._setup_rust_backend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        self._backend = backend
    
    def _setup_numba_backend(self) -> None:
        """Setup the numba backend implementation."""
        # Ensure valid cost function parameters for numba backend
        # If all cost function parameters are -1.0, set default values
        sigma = self.sigma
        theta = self.theta
        beta = self.beta
        mu = self.mu
        
        if (sigma <= 0 and theta <= 0 and beta <= 0 and mu <= 0):
            # Default to simple theta=1 cost function (min load)
            theta = 1.0
        
        self._filter = build_filter_numba(
            universe=self.universe,
            nsubfilters=self.nsubfilters,
            nblocks=self.nblocks,
            nchoices=self.nchoices,
            nbits=self.nbits,
            bitversion=self.bitversion,
            sigma=sigma,
            theta=theta,
            beta=beta,
            mu=mu,
            **self.kwargs
        )
        self._bit_positions_cache = np.zeros(self.nbits, dtype=np.uint64)
    
    def _setup_rust_backend(self) -> None:
        """Setup the rust backend implementation."""
        # For exact parity, we need to use the same hash functions as the numba backend
        # First, create a temporary numba filter to get the exact hash function multipliers
        temp_numba_filter = self._create_temp_numba_filter()
        
        # Extract the multipliers from the numba filter
        subfilter_multipliers = self._extract_multipliers(temp_numba_filter.subfilter_hashfuncs_str)
        bucket_multipliers = self._extract_multipliers(temp_numba_filter.bucket_hashfuncs_str)
        bit_multipliers = self._extract_multipliers(temp_numba_filter.bit_hashfuncs_str)
        
        # Ensure valid cost function parameters for rust backend
        sigma = self.sigma
        theta = self.theta
        beta = self.beta
        mu = self.mu
        
        if (sigma <= 0 and theta <= 0 and beta <= 0 and mu <= 0):
            # Default to simple theta=1 cost function (min load)
            theta = 1.0
        
        # Create the rust filter with the exact same multipliers
        if RustBlowChocExactFilterWithMultipliers is not None:
            rust_filter = RustBlowChocExactFilterWithMultipliers(
                universe=self.universe,
                nsubfilters=self.nsubfilters,
                nblocks=self.nblocks,
                nchoices=self.nchoices,
                nbits=self.nbits,
                bitversion=self.bitversion,
                sigma=sigma,
                theta=theta,
                beta=beta,
                mu=mu,
                subfilter_multipliers=subfilter_multipliers,
                bucket_multipliers=bucket_multipliers,
                bit_multipliers=bit_multipliers,
            )
        else:
            # Fallback to regular exact filter if custom multipliers not available
            rust_filter = RustBlowChocExactFilter(
                universe=self.universe,
                nsubfilters=self.nsubfilters,
                nblocks=self.nblocks,
                nchoices=self.nchoices,
                nbits=self.nbits,
                bitversion=self.bitversion,
                sigma=sigma,
                theta=theta,
                beta=beta,
                mu=mu,
            )
        
        self._filter = rust_filter
    
    def _create_temp_numba_filter(self):
        """Create a temporary numba filter to extract hash function multipliers."""
        # Ensure valid cost function parameters for numba backend
        sigma = self.sigma
        theta = self.theta
        beta = self.beta
        mu = self.mu
        
        if (sigma <= 0 and theta <= 0 and beta <= 0 and mu <= 0):
            # Default to simple theta=1 cost function (min load)
            theta = 1.0
        
        return build_filter_numba(
            universe=self.universe,
            nsubfilters=self.nsubfilters,
            nblocks=self.nblocks,
            nchoices=self.nchoices,
            nbits=self.nbits,
            bitversion=self.bitversion,
            sigma=sigma,
            theta=theta,
            beta=beta,
            mu=mu,
            **self.kwargs
        )
    
    def _extract_multipliers(self, hash_funcs_str):
        """Extract multipliers from hash function strings."""
        multipliers = []
        for func_str in hash_funcs_str:
            if func_str.startswith('linear'):
                multiplier = int(func_str[6:])
                multipliers.append(multiplier)
        return multipliers
    
    def insert(self, key: int, bit_positions: Optional[np.ndarray] = None) -> None:
        """
        Insert a key into the filter.
        
        Args:
            key: Key to insert
            bit_positions: Pre-allocated array for bit positions (optional)
        """
        if self._backend == 'numba':
            if bit_positions is None:
                bit_positions = self._bit_positions_cache
            self._filter.insert(self._filter.array, key, bit_positions)
            self._insertion_count += 1
        elif self._backend == 'rust':
            # Rust implementation doesn't need bit_positions parameter
            self._filter.insert(key)
    
    def lookup(self, key: int, bit_positions: Optional[np.ndarray] = None) -> bool:
        """
        Check if a key might be in the filter.
        
        Args:
            key: Key to lookup
            bit_positions: Pre-allocated array for bit positions (optional)
            
        Returns:
            True if key might be present, False if definitely not present
        """
        if self._backend == 'numba':
            if bit_positions is None:
                bit_positions = self._bit_positions_cache
            return bool(self._filter.lookup(self._filter.array, key, bit_positions))
        elif self._backend == 'rust':
            return self._filter.lookup(key)
    
    def get_fpr(self) -> float:
        """Get the false positive rate of the filter."""
        if self._backend == 'numba':
            return self._filter.get_fpr(self._filter.array)
        elif self._backend == 'rust':
            return self._filter.get_fpr()
    
    def get_fill(self) -> float:
        """Get the fill rate (load factor) of the filter."""
        if self._backend == 'numba':
            return self._filter.get_fill(self._filter.array)
        elif self._backend == 'rust':
            return self._filter.get_fill()
    
    def get_fill_histogram(self) -> np.ndarray:
        """Get histogram of block fill rates."""
        if self._backend == 'numba':
            return self._filter.get_fill_histogram(self._filter.array)
        elif self._backend == 'rust':
            # Return block load distribution
            return np.array(self._filter.get_fill_histogram())
    
    def clear(self) -> None:
        """Clear all bits in the filter."""
        if self._backend == 'numba':
            # For numba backend, recreate the filter
            self._setup_numba_backend()
            self._insertion_count = 0
        elif self._backend == 'rust':
            self._filter.clear()
    
    def stats(self) -> str:
        """Get statistics about the filter."""
        if self._backend == 'numba':
            return (f"BlowChoc Filter (numba):\n"
                   f"  Universe: {self.universe}\n"
                   f"  Subfilters: {self.nsubfilters}\n"
                   f"  Blocks: {self.nblocks}\n"
                   f"  Choices: {self.nchoices}\n"
                   f"  Bits per key: {self.nbits}\n"
                   f"  Bit version: {self.bitversion}\n"
                   f"  Fill rate: {self.get_fill():.4f}\n"
                   f"  FPR: {self.get_fpr():.6f}")
        elif self._backend == 'rust':
            return f"BlowChoc Filter (rust):\n{self._filter.stats()}"
    
    # Properties for compatibility with original API
    @property
    def filtertype(self) -> str:
        """Filter type identifier."""
        return "blowchoc"
    
    @property
    def array(self) -> np.ndarray:
        """Access to underlying bit array (numba backend only)."""
        if self._backend == 'numba':
            return self._filter.array
        else:
            # For rust backend, return block data
            return np.array(self._filter.array())
    
    @property
    def filterbits(self) -> int:
        """Total number of bits in the filter."""
        if self._backend == 'numba':
            return self._filter.filterbits
        else:
            return self._filter.filterbits()
    
    @property
    def mem_bytes(self) -> int:
        """Memory usage in bytes."""
        if self._backend == 'numba':
            return self._filter.mem_bytes
        else:
            return self._filter.mem_bytes()
    
    @property
    def backend(self) -> str:
        """Current backend name."""
        return self._backend
    
    def available_backends(self) -> list[str]:
        """Get list of available backends."""
        backends = []
        if NUMBA_AVAILABLE:
            backends.append('numba')
        if RUST_AVAILABLE:
            backends.append('rust')
        return backends
    
    def current_backend(self) -> str:
        """Get the current backend name."""
        return self._backend
    
    def count(self) -> int:
        """Get the number of elements inserted."""
        if self._backend == 'rust':
            return self._filter.len()
        else:
            # Track insertions manually for numba backend
            return self._insertion_count
    
    # Convenience methods for compatibility
    def contains(self, key: int) -> bool:
        """Alias for lookup (more intuitive name)."""
        return self.lookup(key)
    
    def add(self, key: int) -> None:
        """Alias for insert (more intuitive name)."""
        if self._backend == 'numba':
            self.insert(key)
        elif self._backend == 'rust':
            self._filter.add(key)
    
    def __len__(self) -> int:
        """Return number of elements if available."""
        if self._backend == 'rust':
            return self._filter.len()
        else:
            # Numba backend doesn't track insertion count
            return 0
    
    def __str__(self) -> str:
        return self.stats()
    
    def __repr__(self) -> str:
        return (f"BlowChocFilter(universe={self.universe}, "
                f"nblocks={self.nblocks}, nchoices={self.nchoices}, "
                f"nbits={self.nbits}, backend='{self.backend}')")


# Convenience function that matches original API
def build_filter(*args, backend: str = 'numba', **kwargs) -> BlowChocFilter:
    """
    Build a BlowChoc filter with the unified API.
    
    This function provides the same interface as the original build_filter
    but returns a BlowChocFilter that can switch backends.
    
    Args:
        *args: Positional arguments (universe, nsubfilters, nblocks, nchoices, nbits)
        backend: Backend to use ('numba' or 'rust')
        **kwargs: Additional keyword arguments
        
    Returns:
        BlowChocFilter instance
    """
    if len(args) >= 5:
        universe, nsubfilters, nblocks, nchoices, nbits = args[:5]
        return BlowChocFilter(
            universe=universe,
            nsubfilters=nsubfilters,
            nblocks=nblocks,
            nchoices=nchoices,
            nbits=nbits,
            backend=backend,
            **kwargs
        )
    else:
        return BlowChocFilter(backend=backend, **kwargs)


# For backward compatibility, provide access to original implementation
def build_filter_original(*args, **kwargs):
    """Build filter using only the original numba backend."""
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba backend not available")
    return build_filter_numba(*args, **kwargs)


def build_filter_rust(*args, **kwargs):
    """Build filter using only the rust backend."""
    if not RUST_AVAILABLE:
        raise ImportError("Rust backend not available")
    
    # Convert args to BlowChocFilter parameters
    if len(args) >= 5:
        universe, nsubfilters, nblocks, nchoices, nbits = args[:5]
        filter_obj = BlowChocFilter(
            universe=universe,
            nsubfilters=nsubfilters,
            nblocks=nblocks,
            nchoices=nchoices,
            nbits=nbits,
            backend='rust',
            **kwargs
        )
        return filter_obj
    else:
        return BlowChocFilter(backend='rust', **kwargs) 