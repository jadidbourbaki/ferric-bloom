//! # Ferric Bloom
//!
//! A high-performance blocked bloom filter library implementing the exact BlowChoc algorithm.
//! Provides both Rust and Python APIs with bit-level parity to the original numba implementation.

pub mod blowchoc;

pub use blowchoc::{BitVersion, BlowChocExactFilter};

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
    fn test_blowchoc_exact_filter() {
        let mut filter = BlowChocExactFilter::new(
            4_u64.pow(10), // 4^10 universe
            1,             // 1 subfilter
            100,           // 100 blocks
            2,             // 2 choices
            8,             // 8 bits per key
            BitVersion::Random,
            0.0, // sigma
            1.0, // theta
            0.0, // beta
            0.0, // mu
        )
        .unwrap();

        // Insert some keys
        filter.insert(42);
        filter.insert(1337);
        filter.insert(9999);

        // Test lookups
        assert!(filter.lookup(42));
        assert!(filter.lookup(1337));
        assert!(filter.lookup(9999));

        assert_eq!(filter.len(), 3);
        assert!(!filter.is_empty());
    }
}
