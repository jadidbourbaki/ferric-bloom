//! Standard Bloom filter implementation
//!
//! A space-efficient probabilistic data structure for membership testing.

use crate::{hash::HashFunction, FerricError, Result};
use bit_vec::BitVec;

/// A standard Bloom filter
pub struct BloomFilter {
    /// Bit array storing the filter data
    bits: BitVec,
    /// Hash functions used for this filter
    hash_functions: Vec<Box<dyn HashFunction>>,
    /// Number of elements inserted (for statistics)
    count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    /// * `capacity` - Expected number of elements to insert
    /// * `num_hashes` - Number of hash functions to use
    ///
    /// The bit array size is calculated to achieve approximately 1% false positive rate
    pub fn new(capacity: usize, num_hashes: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FerricError::InvalidParameter(
                "Capacity must be > 0".to_string(),
            ));
        }
        if num_hashes == 0 {
            return Err(FerricError::InvalidParameter(
                "Number of hashes must be > 0".to_string(),
            ));
        }

        // Calculate optimal bit array size for ~1% FPR
        // m = -n * ln(p) / (ln(2))^2, where p is desired FPR
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let bits_needed = (-((capacity as f64) * (0.01_f64).ln()) / ln2_squared) as usize;

        // Round up to next multiple of 64 for efficiency
        let bit_count = ((bits_needed + 63) / 64) * 64;

        let hash_functions = crate::hash::create_hash_functions(num_hashes);

        Ok(BloomFilter {
            bits: BitVec::from_elem(bit_count, false),
            hash_functions,
            count: 0,
        })
    }

    /// Create a Bloom filter with specific parameters
    pub fn with_size(bit_count: usize, num_hashes: usize) -> Result<Self> {
        if bit_count == 0 {
            return Err(FerricError::InvalidParameter(
                "Bit count must be > 0".to_string(),
            ));
        }
        if num_hashes == 0 {
            return Err(FerricError::InvalidParameter(
                "Number of hashes must be > 0".to_string(),
            ));
        }

        let hash_functions = crate::hash::create_hash_functions(num_hashes);

        Ok(BloomFilter {
            bits: BitVec::from_elem(bit_count, false),
            hash_functions,
            count: 0,
        })
    }

    /// Insert a key into the filter
    pub fn insert(&mut self, key: u64) {
        let bit_count = self.bits.len() as u64;

        for hash_fn in &self.hash_functions {
            let hash_val = hash_fn.hash(key, bit_count);
            self.bits.set(hash_val as usize, true);
        }

        self.count += 1;
    }

    /// Check if a key might be in the filter
    /// Returns true if the key might be present (with possible false positives)
    /// Returns false if the key is definitely not present
    pub fn contains(&self, key: u64) -> bool {
        let bit_count = self.bits.len() as u64;

        for hash_fn in &self.hash_functions {
            let hash_val = hash_fn.hash(key, bit_count);
            if !self.bits.get(hash_val as usize).unwrap_or(false) {
                return false;
            }
        }

        true
    }

    /// Get the current load factor (fraction of bits set)
    pub fn load_factor(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&bit| bit).count();
        set_bits as f64 / self.bits.len() as f64
    }

    /// Get the estimated false positive rate
    pub fn estimated_fpr(&self) -> f64 {
        let load = self.load_factor();
        load.powi(self.hash_functions.len() as i32)
    }

    /// Get statistics about the filter
    pub fn stats(&self) -> BloomStats {
        BloomStats {
            capacity: self.bits.len(),
            num_hash_functions: self.hash_functions.len(),
            elements_inserted: self.count,
            load_factor: self.load_factor(),
            estimated_fpr: self.estimated_fpr(),
        }
    }

    /// Clear all bits in the filter
    pub fn clear(&mut self) {
        self.bits.clear();
        self.bits.grow(self.bits.len(), false);
        self.count = 0;
    }

    /// Get the number of elements inserted
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the capacity (number of bits)
    pub fn capacity(&self) -> usize {
        self.bits.len()
    }

    /// Get the number of hash functions
    pub fn num_hash_functions(&self) -> usize {
        self.hash_functions.len()
    }
}

/// Statistics about a Bloom filter
#[derive(Debug, Clone)]
pub struct BloomStats {
    pub capacity: usize,
    pub num_hash_functions: usize,
    pub elements_inserted: usize,
    pub load_factor: f64,
    pub estimated_fpr: f64,
}

impl std::fmt::Display for BloomStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "BloomFilter Stats:\n\
             - Capacity: {} bits\n\
             - Hash functions: {}\n\
             - Elements inserted: {}\n\
             - Load factor: {:.3}\n\
             - Estimated FPR: {:.6}",
            self.capacity,
            self.num_hash_functions,
            self.elements_inserted,
            self.load_factor,
            self.estimated_fpr
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bloom = BloomFilter::new(1000, 3).unwrap();

        // Test insertion and lookup
        bloom.insert(42);
        bloom.insert(1337);
        bloom.insert(9999);

        assert!(bloom.contains(42));
        assert!(bloom.contains(1337));
        assert!(bloom.contains(9999));

        // The load factor should be > 0 after insertions
        assert!(bloom.load_factor() > 0.0);
    }

    #[test]
    fn test_bloom_filter_false_negatives() {
        let mut bloom = BloomFilter::new(1000, 3).unwrap();

        // Item not inserted should definitely not be found
        // (though false positives are possible for other items)
        let test_key = 99999;
        assert!(!bloom.contains(test_key));

        // After insertion, it should definitely be found
        bloom.insert(test_key);
        assert!(bloom.contains(test_key));
    }

    #[test]
    fn test_bloom_filter_stats() {
        let mut bloom = BloomFilter::new(1000, 5).unwrap();

        for i in 0..100 {
            bloom.insert(i);
        }

        let stats = bloom.stats();
        assert_eq!(stats.num_hash_functions, 5);
        assert_eq!(stats.elements_inserted, 100);
        assert!(stats.load_factor > 0.0);
        assert!(stats.estimated_fpr > 0.0);
    }

    #[test]
    fn test_bloom_filter_clear() {
        let mut bloom = BloomFilter::new(1000, 3).unwrap();

        bloom.insert(42);
        assert!(bloom.contains(42));

        bloom.clear();
        // After clear, the item should not be found
        // (though there might be false positives for other keys)
        assert_eq!(bloom.load_factor(), 0.0);
        assert_eq!(bloom.stats().elements_inserted, 0);
    }
}
