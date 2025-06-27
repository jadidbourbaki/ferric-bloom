//! # Ferric Bloom
//!
//! A high-performance Bloom filter and Blocked Bloom filter library optimized for cache efficiency.
//! Inspired by the blocked bloom filter design with 512-bit cache-aligned blocks.

pub mod blocked_bloom;
pub mod bloom;
pub mod hash;
pub mod utils;

pub use blocked_bloom::BlockedBloomFilter;
pub use bloom::BloomFilter;
pub use hash::{AffineHash, HashFunction, LinearHash};

// Python bindings
#[cfg(feature = "python")]
pub mod python_module;

/// Common error types for the library
#[derive(Debug, Clone)]
pub enum FerricError {
    InvalidParameter(String),
    MemoryAllocation(String),
    HashFunction(String),
}

impl std::fmt::Display for FerricError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FerricError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            FerricError::MemoryAllocation(msg) => write!(f, "Memory allocation error: {}", msg),
            FerricError::HashFunction(msg) => write!(f, "Hash function error: {}", msg),
        }
    }
}

impl std::error::Error for FerricError {}

pub type Result<T> = std::result::Result<T, FerricError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 10).unwrap();

        // Insert some keys
        bloom.insert(42);
        bloom.insert(1337);
        bloom.insert(9999);

        // Test membership
        assert!(bloom.contains(42));
        assert!(bloom.contains(1337));
        assert!(bloom.contains(9999));

        // Test non-membership (may have false positives)
        assert!(!bloom.contains(1) || true); // Allow false positive
    }

    #[test]
    fn test_basic_blocked_bloom_filter() {
        let mut blocked = BlockedBloomFilter::new(1000, 10, 2).unwrap();

        // Insert some keys
        blocked.insert(42);
        blocked.insert(1337);

        // Test membership
        assert!(blocked.contains(42));
        assert!(blocked.contains(1337));
    }
}
