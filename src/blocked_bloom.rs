//! Blocked Bloom filter implementation
//!
//! Implements a cache-efficient blocked Bloom filter with 512-bit blocks,
//! similar to the design in blowchoc.

use crate::{hash::HashFunction, FerricError, Result};

/// Size of each block in bits (512 bits = 64 bytes = 1 cache line)
pub const BLOCK_SIZE: usize = 512;

/// Number of u64 words per block
const WORDS_PER_BLOCK: usize = BLOCK_SIZE / 64; // 8 words

/// A blocked Bloom filter optimized for cache efficiency
pub struct BlockedBloomFilter {
    /// Blocks of bits, each aligned to cache lines
    blocks: Vec<[u64; WORDS_PER_BLOCK]>,
    /// Number of choices (alternative blocks) to consider for each key
    num_choices: usize,
    /// Number of bits to set per key within a block
    bits_per_key: usize,
    /// Hash functions for selecting blocks
    block_hash_functions: Vec<Box<dyn HashFunction>>,
    /// Hash functions for selecting bit positions within blocks
    bit_hash_functions: Vec<Box<dyn HashFunction>>,
    /// Cost function for selecting among alternative blocks
    cost_function: CostFunction,
    /// Statistics
    count: usize,
}

/// Cost function types for selecting blocks
#[derive(Debug, Clone)]
pub enum CostFunction {
    /// Select block with minimum number of set bits
    MinLoad,
    /// Select block with minimum expected FPR after insertion
    MinFpr { sigma: f64, theta: f64 },
    /// Look-ahead cost function
    LookAhead { mu: f64 },
}

impl BlockedBloomFilter {
    /// Create a new blocked Bloom filter
    ///
    /// # Arguments
    /// * `num_blocks` - Number of 512-bit blocks
    /// * `bits_per_key` - Number of bits to set per key (like k in traditional Bloom filters)
    /// * `num_choices` - Number of alternative blocks to consider (1 = no choice, 2+ = choices)
    pub fn new(num_blocks: usize, bits_per_key: usize, num_choices: usize) -> Result<Self> {
        if num_blocks == 0 {
            return Err(FerricError::InvalidParameter(
                "Number of blocks must be > 0".to_string(),
            ));
        }
        if bits_per_key == 0 || bits_per_key > BLOCK_SIZE {
            return Err(FerricError::InvalidParameter(format!(
                "Bits per key must be 1-{}",
                BLOCK_SIZE
            )));
        }
        if num_choices == 0 {
            return Err(FerricError::InvalidParameter(
                "Number of choices must be > 0".to_string(),
            ));
        }

        // Create aligned blocks (Vec automatically aligns, but we could use aligned allocator)
        let blocks = vec![[0u64; WORDS_PER_BLOCK]; num_blocks];

        // Create hash functions for block selection
        let block_hash_functions = crate::hash::create_hash_functions(num_choices);

        // Create hash functions for bit position selection within blocks
        let bit_hash_functions = crate::hash::create_hash_functions(bits_per_key);

        Ok(BlockedBloomFilter {
            blocks,
            num_choices,
            bits_per_key,
            block_hash_functions,
            bit_hash_functions,
            cost_function: CostFunction::MinLoad,
            count: 0,
        })
    }

    /// Set the cost function for block selection
    pub fn set_cost_function(&mut self, cost_function: CostFunction) {
        self.cost_function = cost_function;
    }

    /// Insert a key into the filter
    pub fn insert(&mut self, key: u64) {
        // Get bit positions within a block
        let bit_positions = self.get_bit_positions(key);

        // Find the best block to insert into
        let best_block_idx = self.find_best_block(key, &bit_positions);

        // Set the bits in the selected block
        self.set_bits_in_block(best_block_idx, &bit_positions);

        self.count += 1;
    }

    /// Check if a key might be in the filter
    pub fn contains(&self, key: u64) -> bool {
        let bit_positions = self.get_bit_positions(key);

        // Check all possible blocks for this key
        for choice in 0..self.num_choices {
            let block_idx = self.get_block_index(key, choice);
            if self.check_bits_in_block(block_idx, &bit_positions) {
                return true;
            }
        }

        false
    }

    /// Get bit positions within a block for a given key
    fn get_bit_positions(&self, key: u64) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.bits_per_key);

        for (i, hash_fn) in self.bit_hash_functions.iter().enumerate() {
            // Use different variations of the key for each hash function
            let varied_key = key.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15)); // Golden ratio
            let pos = hash_fn.hash(varied_key, BLOCK_SIZE as u64) as usize;
            positions.push(pos);
        }

        // Ensure distinct positions (similar to blowchoc's "distinct" mode)
        positions.sort_unstable();
        positions.dedup();

        // If we lost positions due to deduplication, generate more
        while positions.len() < self.bits_per_key {
            let extra_key = key.wrapping_mul(positions.len() as u64 + 1);
            let pos = (extra_key % BLOCK_SIZE as u64) as usize;
            if !positions.contains(&pos) {
                positions.push(pos);
            }
        }

        positions
    }

    /// Find the best block to insert a key into
    fn find_best_block(&self, key: u64, bit_positions: &[usize]) -> usize {
        if self.num_choices == 1 {
            return self.get_block_index(key, 0);
        }

        let mut best_block = 0;
        let mut best_cost = f64::INFINITY;

        for choice in 0..self.num_choices {
            let block_idx = self.get_block_index(key, choice);
            let cost = self.calculate_block_cost(block_idx, bit_positions);

            if cost < best_cost {
                best_cost = cost;
                best_block = block_idx;
            }
        }

        best_block
    }

    /// Calculate the cost of inserting into a specific block
    fn calculate_block_cost(&self, block_idx: usize, bit_positions: &[usize]) -> f64 {
        let current_load = self.count_set_bits_in_block(block_idx);
        let new_bits = self.count_new_bits_in_block(block_idx, bit_positions);

        match self.cost_function {
            CostFunction::MinLoad => current_load as f64,
            CostFunction::MinFpr { sigma, theta } => {
                let q = current_load + new_bits;
                let local_fpr = (q as f64 / BLOCK_SIZE as f64).powi(self.bits_per_key as i32);
                sigma * local_fpr + theta * (new_bits as f64)
            }
            CostFunction::LookAhead { mu } => {
                let q = current_load + new_bits + (mu * self.bits_per_key as f64) as usize;
                let future_fpr = (q as f64 / BLOCK_SIZE as f64).powi(self.bits_per_key as i32);
                future_fpr + (new_bits as f64)
            }
        }
    }

    /// Get block index for a key and choice
    fn get_block_index(&self, key: u64, choice: usize) -> usize {
        let hash_val = self.block_hash_functions[choice].hash(key, self.blocks.len() as u64);
        hash_val as usize
    }

    /// Clear all bits in the filter
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.fill(0);
        }
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

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the block size in bits
    pub fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    /// Get the number of bits per key
    pub fn bits_per_key(&self) -> usize {
        self.bits_per_key
    }

    /// Get the number of choices
    pub fn num_choices(&self) -> usize {
        self.num_choices
    }

    /// Get estimated false positive rate
    pub fn estimated_fpr(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let stats = self.stats();
        stats.estimated_fpr
    }

    /// Get current load factor
    pub fn load_factor(&self) -> f64 {
        let stats = self.stats();
        stats.load_factor
    }

    /// Get raw block data for Python access
    pub fn get_blocks(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.blocks.len() * WORDS_PER_BLOCK);
        for block in &self.blocks {
            result.extend_from_slice(block);
        }
        result
    }

    /// Get block load distribution
    pub fn get_block_loads(&self) -> Vec<usize> {
        self.blocks
            .iter()
            .map(|block| block.iter().map(|word| word.count_ones() as usize).sum())
            .collect()
    }

    /// Set bits in a specific block
    fn set_bits_in_block(&mut self, block_idx: usize, bit_positions: &[usize]) {
        for &pos in bit_positions {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            self.blocks[block_idx][word_idx] |= 1u64 << bit_idx;
        }
    }

    /// Check if all bits are set in a specific block
    fn check_bits_in_block(&self, block_idx: usize, bit_positions: &[usize]) -> bool {
        for &pos in bit_positions {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            if (self.blocks[block_idx][word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Count set bits in a block
    fn count_set_bits_in_block(&self, block_idx: usize) -> usize {
        self.blocks[block_idx]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Count how many new bits would be set
    fn count_new_bits_in_block(&self, block_idx: usize, bit_positions: &[usize]) -> usize {
        let mut new_bits = 0;
        for &pos in bit_positions {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            if (self.blocks[block_idx][word_idx] & (1u64 << bit_idx)) == 0 {
                new_bits += 1;
            }
        }
        new_bits
    }

    /// Get statistics about the filter
    pub fn stats(&self) -> BlockedBloomStats {
        let total_bits = self.blocks.len() * BLOCK_SIZE;
        let set_bits: usize = self
            .blocks
            .iter()
            .map(|block| {
                block
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>()
            })
            .sum();

        let load_factor = set_bits as f64 / total_bits as f64;
        let estimated_fpr = load_factor.powi(self.bits_per_key as i32) * self.num_choices as f64;

        // Calculate block load distribution
        let block_loads: Vec<usize> = self
            .blocks
            .iter()
            .map(|block| block.iter().map(|word| word.count_ones() as usize).sum())
            .collect();

        let avg_block_load = block_loads.iter().sum::<usize>() as f64 / block_loads.len() as f64;
        let max_block_load = *block_loads.iter().max().unwrap_or(&0);
        let min_block_load = *block_loads.iter().min().unwrap_or(&0);

        BlockedBloomStats {
            num_blocks: self.blocks.len(),
            block_size: BLOCK_SIZE,
            num_choices: self.num_choices,
            bits_per_key: self.bits_per_key,
            elements_inserted: self.count,
            total_bits,
            set_bits,
            load_factor,
            estimated_fpr,
            avg_block_load,
            min_block_load,
            max_block_load,
        }
    }
}

/// Statistics about a Blocked Bloom filter
#[derive(Debug, Clone)]
pub struct BlockedBloomStats {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_choices: usize,
    pub bits_per_key: usize,
    pub elements_inserted: usize,
    pub total_bits: usize,
    pub set_bits: usize,
    pub load_factor: f64,
    pub estimated_fpr: f64,
    pub avg_block_load: f64,
    pub min_block_load: usize,
    pub max_block_load: usize,
}

impl std::fmt::Display for BlockedBloomStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "BlockedBloomFilter Stats:\n\
             - Blocks: {} × {} bits\n\
             - Choices: {}, Bits per key: {}\n\
             - Elements inserted: {}\n\
             - Load factor: {:.3}\n\
             - Estimated FPR: {:.6}\n\
             - Block load: avg={:.1}, min={}, max={}",
            self.num_blocks,
            self.block_size,
            self.num_choices,
            self.bits_per_key,
            self.elements_inserted,
            self.load_factor,
            self.estimated_fpr,
            self.avg_block_load,
            self.min_block_load,
            self.max_block_load
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_bloom_basic() {
        let mut blocked = BlockedBloomFilter::new(10, 5, 2).unwrap();

        // Insert some keys
        blocked.insert(42);
        blocked.insert(1337);
        blocked.insert(9999);

        // Test membership
        assert!(blocked.contains(42));
        assert!(blocked.contains(1337));
        assert!(blocked.contains(9999));
    }

    #[test]
    fn test_blocked_bloom_cost_functions() {
        let mut blocked = BlockedBloomFilter::new(10, 3, 2).unwrap();

        // Test different cost functions
        blocked.set_cost_function(CostFunction::MinLoad);
        blocked.insert(42);

        blocked.set_cost_function(CostFunction::MinFpr {
            sigma: 1.0,
            theta: 1.0,
        });
        blocked.insert(1337);

        blocked.set_cost_function(CostFunction::LookAhead { mu: 0.5 });
        blocked.insert(9999);

        // All should be findable
        assert!(blocked.contains(42));
        assert!(blocked.contains(1337));
        assert!(blocked.contains(9999));
    }

    #[test]
    fn test_blocked_bloom_stats() {
        let mut blocked = BlockedBloomFilter::new(5, 4, 2).unwrap();

        for i in 0..20 {
            blocked.insert(i);
        }

        let stats = blocked.stats();
        assert_eq!(stats.num_blocks, 5);
        assert_eq!(stats.block_size, BLOCK_SIZE);
        assert_eq!(stats.num_choices, 2);
        assert_eq!(stats.bits_per_key, 4);
        assert_eq!(stats.elements_inserted, 20);
        assert!(stats.load_factor > 0.0);
    }
}
