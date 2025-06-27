//! Hash functions for Bloom filters
//!
//! Implements linear and affine hash functions similar to those used in blowchoc.

use crate::{FerricError, Result};

/// Trait for hash functions used in Bloom filters
pub trait HashFunction: Send + Sync {
    /// Hash a key with the given modulus
    fn hash(&self, key: u64, modulus: u64) -> u64;

    /// Get a name/identifier for this hash function
    fn name(&self) -> String;
}

/// Linear hash function: h(x) = (a * x) mod m
/// Where 'a' must be odd for the function to be bijective
#[derive(Debug, Clone)]
pub struct LinearHash {
    multiplier: u64,
}

impl LinearHash {
    /// Create a new linear hash function with the given multiplier
    /// The multiplier must be odd to ensure bijectivity
    pub fn new(multiplier: u64) -> Result<Self> {
        if multiplier % 2 == 0 {
            return Err(FerricError::HashFunction(
                "Linear hash multiplier must be odd".to_string(),
            ));
        }
        Ok(LinearHash { multiplier })
    }

    /// Create a linear hash with a random odd multiplier
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let multiplier = rng.gen_range(3..=u64::MAX) | 1; // Ensure odd
        LinearHash { multiplier }
    }
}

impl HashFunction for LinearHash {
    fn hash(&self, key: u64, modulus: u64) -> u64 {
        if modulus == 0 {
            return 0;
        }

        // For DNA k-mers, we implement the bit-swapping optimization from blowchoc
        let qbits = 64 - key.leading_zeros() as usize;
        if qbits % 2 == 0 && qbits > 0 && qbits < 64 {
            let q = qbits / 2;
            let mask = (1u64 << qbits) - 1;
            let swapped = ((key << q) ^ (key >> q)) & mask;
            (self.multiplier.wrapping_mul(swapped)) % modulus
        } else {
            (self.multiplier.wrapping_mul(key)) % modulus
        }
    }

    fn name(&self) -> String {
        format!("linear{}", self.multiplier)
    }
}

/// Affine hash function: h(x) = (a * x + b) mod m
#[derive(Debug, Clone)]
pub struct AffineHash {
    multiplier: u64,
    addend: u64,
}

impl AffineHash {
    /// Create a new affine hash function
    pub fn new(multiplier: u64, addend: u64) -> Result<Self> {
        if multiplier % 2 == 0 {
            return Err(FerricError::HashFunction(
                "Affine hash multiplier must be odd".to_string(),
            ));
        }
        Ok(AffineHash { multiplier, addend })
    }

    /// Create an affine hash with random parameters
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let multiplier = rng.gen_range(3..=u64::MAX) | 1; // Ensure odd
        let addend = rng.gen();
        AffineHash { multiplier, addend }
    }
}

impl HashFunction for AffineHash {
    fn hash(&self, key: u64, modulus: u64) -> u64 {
        if modulus == 0 {
            return 0;
        }

        let hash_val = self.multiplier.wrapping_mul(key).wrapping_add(self.addend);
        hash_val % modulus
    }

    fn name(&self) -> String {
        format!("affine{}-{}", self.multiplier, self.addend)
    }
}

/// Create a set of diverse hash functions for use in Bloom filters
pub fn create_hash_functions(count: usize) -> Vec<Box<dyn HashFunction>> {
    let mut functions: Vec<Box<dyn HashFunction>> = Vec::with_capacity(count);

    // Use some predefined good multipliers from blowchoc
    let good_multipliers = [62591, 42953, 48271, 69621, 83791, 91019];

    for i in 0..count {
        if i < good_multipliers.len() {
            functions
                .push(Box::new(LinearHash::new(good_multipliers[i]).unwrap())
                    as Box<dyn HashFunction>);
        } else {
            functions.push(Box::new(LinearHash::random()) as Box<dyn HashFunction>);
        }
    }

    functions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_hash() {
        let hasher = LinearHash::new(123).unwrap();

        // Test basic functionality - just ensure it produces valid results
        let result = hasher.hash(42, 100);
        assert!(result < 100);
        assert_eq!(hasher.hash(0, 100), 0);

        // Test edge cases
        assert_eq!(hasher.hash(42, 0), 0);
        assert_eq!(hasher.hash(42, 1), 0);
    }

    #[test]
    fn test_linear_hash_odd_multiplier() {
        assert!(LinearHash::new(124).is_err()); // Even multiplier should fail
        assert!(LinearHash::new(123).is_ok()); // Odd multiplier should work
    }

    #[test]
    fn test_affine_hash() {
        let hasher = AffineHash::new(123, 456).unwrap();

        // Test basic functionality
        let expected = (123u64.wrapping_mul(42).wrapping_add(456)) % 100;
        assert_eq!(hasher.hash(42, 100), expected);
    }

    #[test]
    fn test_hash_function_diversity() {
        let functions = create_hash_functions(5);
        assert_eq!(functions.len(), 5);

        // Test that different functions produce different results for the same input
        let key = 42u64;
        let modulus = 1000u64;
        let results: Vec<u64> = functions.iter().map(|f| f.hash(key, modulus)).collect();

        // Should have some diversity (not all the same)
        let unique_results: std::collections::HashSet<_> = results.iter().collect();
        assert!(unique_results.len() > 1);
    }
}
