"""
Ferric Bloom - High-performance Bloom filters implemented in Rust

This package provides Python bindings for the ferric-bloom Rust library,
offering both standard and blocked Bloom filters with excellent performance.
"""

try:
    from .ferric_bloom import (
        # Main classes
        BloomFilter,
        BlockedBloomFilter,
        CostFunction,
        
        # Utility functions
        py_optimal_bloom_parameters as optimal_bloom_parameters,
        py_optimal_blocked_bloom_parameters as optimal_blocked_bloom_parameters,
        py_create_hash_multipliers as create_hash_multipliers,
        
        # DNA utilities
        encode_dna_kmer,
        decode_dna_kmer,
        
        # Benchmarking
        benchmark_bloom_filter,
        benchmark_blocked_bloom_filter,
        
        # Constants
        BLOCK_SIZE,
        __version__,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    BloomFilter = None
    BlockedBloomFilter = None
    CostFunction = None
    optimal_bloom_parameters = None
    optimal_blocked_bloom_parameters = None
    create_hash_multipliers = None
    encode_dna_kmer = None
    decode_dna_kmer = None
    benchmark_bloom_filter = None
    benchmark_blocked_bloom_filter = None
    BLOCK_SIZE = 512
    __version__ = "0.1.0"

__all__ = [
    "BloomFilter",
    "BlockedBloomFilter", 
    "CostFunction",
    "optimal_bloom_parameters",
    "optimal_blocked_bloom_parameters",
    "create_hash_multipliers",
    "encode_dna_kmer",
    "decode_dna_kmer",
    "benchmark_bloom_filter",
    "benchmark_blocked_bloom_filter",
    "BLOCK_SIZE",
    "RUST_AVAILABLE",
    "__version__",
] 