//! Utility functions for Bloom filters

/// Calculate optimal Bloom filter parameters
pub struct BloomParameters {
    pub optimal_num_bits: usize,
    pub optimal_num_hashes: usize,
    pub expected_fpr: f64,
}

/// Calculate optimal Bloom filter parameters for given constraints
pub fn optimal_bloom_parameters(
    expected_elements: usize,
    desired_fpr: f64,
    max_memory_bits: Option<usize>,
) -> BloomParameters {
    if expected_elements == 0 {
        return BloomParameters {
            optimal_num_bits: 1,
            optimal_num_hashes: 1,
            expected_fpr: 0.0,
        };
    }

    let n = expected_elements as f64;
    let p = desired_fpr;

    // Optimal number of bits: m = -n * ln(p) / (ln(2))^2
    let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    let optimal_bits = (-n * p.ln() / ln2_squared).ceil() as usize;

    // Apply memory constraint if given
    let final_bits = match max_memory_bits {
        Some(max_bits) if optimal_bits > max_bits => max_bits,
        _ => optimal_bits,
    };

    // Optimal number of hash functions: k = (m/n) * ln(2)
    let m = final_bits as f64;
    let optimal_hashes = ((m / n) * std::f64::consts::LN_2).round() as usize;
    let final_hashes = optimal_hashes.max(1).min(20); // Reasonable bounds

    // Calculate actual FPR with these parameters
    let actual_fpr = (1.0 - (-(final_hashes as f64) * n / m).exp()).powi(final_hashes as i32);

    BloomParameters {
        optimal_num_bits: final_bits,
        optimal_num_hashes: final_hashes,
        expected_fpr: actual_fpr,
    }
}

/// Calculate optimal blocked Bloom filter parameters
pub fn optimal_blocked_bloom_parameters(
    expected_elements: usize,
    desired_fpr: f64,
    num_choices: usize,
) -> (usize, usize) {
    // (num_blocks, bits_per_key)
    let _n = expected_elements as f64;
    let p = desired_fpr / num_choices as f64; // Adjust for multiple choices

    // For blocked filters, we optimize for block utilization
    // Target load factor of ~0.5 per block for good performance
    let target_load = 0.5;
    let block_capacity = (super::blocked_bloom::BLOCK_SIZE as f64 * target_load) as usize;

    let num_blocks = ((expected_elements as f64 / block_capacity as f64).ceil() as usize).max(1);

    // Calculate bits per key for desired FPR
    let bits_per_key = if p > 0.0 {
        (-p.ln() / std::f64::consts::LN_2).ceil() as usize
    } else {
        10 // Default reasonable value
    };

    (num_blocks, bits_per_key.max(1).min(32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_bloom_parameters() {
        let params = optimal_bloom_parameters(1000, 0.01, None);

        assert!(params.optimal_num_bits > 0);
        assert!(params.optimal_num_hashes > 0);
        assert!(params.expected_fpr > 0.0);
        assert!(params.expected_fpr <= 0.02); // Should be close to desired 0.01
    }

    #[test]
    fn test_optimal_bloom_parameters_with_memory_limit() {
        let params = optimal_bloom_parameters(1000, 0.01, Some(1000));

        assert!(params.optimal_num_bits <= 1000);
        assert!(params.optimal_num_hashes > 0);
    }

    #[test]
    fn test_optimal_blocked_bloom_parameters() {
        let (num_blocks, bits_per_key) = optimal_blocked_bloom_parameters(1000, 0.01, 2);

        assert!(num_blocks > 0);
        assert!(bits_per_key > 0);
        assert!(bits_per_key <= 32);
    }
}
