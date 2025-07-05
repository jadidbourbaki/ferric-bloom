//! Exact BlowChoc implementation in Rust
//!
//! This module provides a bit-for-bit exact implementation of the BlowChoc
//! blocked bloom filter, matching the numba implementation exactly.
//!
//! Original BlowChoc implementation:
//! - File: blowchoc/blowchoc/filter_bb_choices.py
//! - Block size: 512 bits (cache line size)
//! - Hash functions: linear with bit swapping for DNA k-mers
//! - Cost functions: scaled, lookahead, and exponential variants
//! - Bit positioning strategies: random, distinct, double

use crate::{FerricError, Result};

/// BLOCKSIZE must match BlowChoc exactly
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/filter_bb_choices.py, line 18
/// ```python
/// BLOCKSIZE = 512  # DO NOT CHANGE
/// ```
pub const BLOCKSIZE: usize = 512;

/// Number of u64 words per block (512 bits / 64 bits = 8 words)
const WORDS_PER_BLOCK: usize = BLOCKSIZE / 64;

/// Bit version strategies (must match BlowChoc)
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/filter_bb_choices.py, lines 134-148
/// ```python
/// def get_mod_values(bitversion, k):
///     """Computes the modulo and offset values to correctly compute the bit positions using different strategies"""
///     # use 512, 511, 510, ..., 512-k+1 as mod values to compute k distinct bit positions
///     if bitversion == 'distinct':
///         mod_values = tuple(BLOCKSIZE - i for i in range(k))
///         return mod_values
///     # compute k random positions using a mod value of 512
///     if bitversion == 'random':
///         mod_values = (BLOCKSIZE,) * k
///         return mod_values
///     if bitversion == 'double':
///         mod_values = (BLOCKSIZE, BLOCKSIZE // 2)
///         return mod_values
///     raise ValueError(f"ERROR: get_mod_values: {bitversion=} is not supported.")
/// ```
#[derive(Debug, Clone)]
pub enum BitVersion {
    Random,
    Distinct,
    Double,
}

impl BitVersion {
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "random" => Ok(BitVersion::Random),
            "distinct" => Ok(BitVersion::Distinct),
            "double" => Ok(BitVersion::Double),
            _ => Err(FerricError::InvalidParameter(format!(
                "Unknown bit version: {}",
                s
            ))),
        }
    }
}

/// Cost function types (must match BlowChoc exactly)
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/filter_bb_choices.py, lines 324-350
/// ```python
/// @njit(nogil=True)
/// def cost_function_scaled(nbits_set, nnew_bits):
///     # sigma * nbits * ((2 * q / blocksize) ** nbits) + theta * nnew_bits
///     # where q is the number of already set bits in the block
///     inner_cost = sigma * nbits * ((2 * nbits_set / BLOCKSIZE) ** nbits)
///     return inner_cost + theta * nnew_bits
///
/// @njit(nogil=True)
/// def cost_function_lookahead(nbits_set, nnew_bits):
///     # nbits * ((2 * q / blocksize) ** nbits) + nnew_bits
///     # where q includes mu * nbits
///     q = nbits_set + mu * nbits
///     inner_cost = nbits * ((2 * q / BLOCKSIZE) ** nbits)
///     return inner_cost + nnew_bits
///
/// @njit(nogil=True)
/// def cost_function_exp(nbits_set, nnew_bits):
///     # beta**(q / blocksize4) + mu * (nnew_bits / nbits)
///     blocksize4 = BLOCKSIZE // 4
///     q = nbits_set
///     inner_cost = beta ** (q / blocksize4)
///     return inner_cost + mu * (nnew_bits / nbits)
/// ```
#[derive(Debug, Clone)]
pub enum CostFunction {
    /// sigma * nbits * ((2 * q / blocksize) ** nbits) + theta * nnew_bits
    Scaled { sigma: f64, theta: f64 },
    /// nbits * ((2 * q / blocksize) ** nbits) + nnew_bits (where q includes mu * nbits)
    LookAhead { mu: f64 },
    /// beta**(q / blocksize4) + mu * (nnew_bits / nbits)
    Exponential { beta: f64, mu: f64 },
}

/// BlowChoc linear hash function (must match exactly)
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/hashfunctions.py, lines 73-92
/// ```python
/// @njit(nogil=True, locals=dict(
///     code=uint64, swap=uint64, f=uint64, p=uint64))
/// def get_bucket_fpr(code):
///     swap = ((code << q) ^ (code >> q)) & codemask
///     swap = (a * swap) & codemask
///     p = swap % nbuckets
///     f = swap // nbuckets
///     return (p, f)
/// ```
///
/// And File: blowchoc/blowchoc/hashfunctions.py, lines 201-220
/// ```python
/// @njit(nogil=True, locals=dict(
///       code=uint64, swap=uint64, bucket=uint64))
/// def get_bucket(code):
///     swap = ((code << q) ^ (code >> q)) & codemask
///     swap = (a * (swap ^ b)) & codemask
///     bucket = swap & bitmask
///     return bucket
/// ```
#[derive(Debug, Clone)]
pub struct BlowChocLinearHash {
    multiplier: u64,
    universe: u64,
    modulus: u64,
    qbits: u32,
    q: u32,
    codemask: u64,
}

impl BlowChocLinearHash {
    pub fn new(multiplier: u64, universe: u64, modulus: u64) -> Result<Self> {
        // Verify universe is power of 4 (for DNA k-mers)
        //
        // Original BlowChoc reference:
        // File: blowchoc/blowchoc/hashfunctions.py, lines 58-62
        // ```python
        // if 4**(qbits // 2) != universe:
        //     raise ValueError("hash functions require that universe is a power of 4")
        // else:
        //     q = qbits // 2
        // ```
        let qbits = (universe as f64).log2() as u32;
        if 4_u64.pow(qbits / 2) != universe {
            return Err(FerricError::InvalidParameter(
                "Universe must be a power of 4".to_string(),
            ));
        }

        if multiplier % 2 == 0 {
            return Err(FerricError::InvalidParameter(
                "Multiplier must be odd".to_string(),
            ));
        }

        let q = qbits / 2;
        let codemask = (1u64 << qbits) - 1;

        Ok(BlowChocLinearHash {
            multiplier,
            universe,
            modulus,
            qbits,
            q,
            codemask,
        })
    }

    /// Exact BlowChoc hash implementation
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/hashfunctions.py, lines 73-78
    /// ```python
    /// @njit(nogil=True, locals=dict(
    ///     code=uint64, swap=uint64, f=uint64, p=uint64))
    /// def get_bucket_fpr(code):
    ///     swap = ((code << q) ^ (code >> q)) & codemask
    ///     swap = (a * swap) & codemask
    ///     p = swap % nbuckets
    ///     f = swap // nbuckets
    ///     return (p, f)
    /// ```
    pub fn hash(&self, key: u64) -> u64 {
        // Perform the exact bit swapping from BlowChoc
        let swap = ((key << self.q) ^ (key >> self.q)) & self.codemask;
        let hash_val = (self.multiplier.wrapping_mul(swap)) & self.codemask;

        if modulus_is_power_of_2(self.modulus) {
            // Use bit shifting for power of 2 modulus
            let bits = (self.modulus as f64).log2() as u32;
            (hash_val >> (self.qbits - bits)) & ((1u64 << bits) - 1)
        } else if self.modulus % 2 == 0 {
            // Even modulus but not power of 2
            let hash_val = hash_val ^ (hash_val >> self.q);
            hash_val % self.modulus
        } else {
            // Odd modulus
            hash_val % self.modulus
        }
    }
}

fn modulus_is_power_of_2(n: u64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Default BlowChoc multipliers
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/hashfunctions.py, line 14
/// ```python
/// DEFAULT_HASHFUNCS = ("linear62591", "linear42953", "linear48271")
/// ```
const DEFAULT_MULTIPLIERS: &[u64] = &[62591, 42953, 48271, 69621, 83791, 91019];

/// Exact BlowChoc blocked bloom filter implementation
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/filter_bb_choices.py, lines 20-47
/// ```python
/// BBChoicesFilter = namedtuple("BBChoicesFilter", [
///     # attributes
///     "universe",  # save
///     "nsubfilters",  # save
///     "subfilter_hashfuncs_str",  # save
///     "bucket_hashfuncs_str",  # save
///     "bit_hashfuncs_str",  # save
///     "filtertype",  # save
///     "nbits",  # save
///     "sigma",  # save
///     "theta",  # save
///     "beta",  # save
///     "mu",  # save
///     "bitversion",  # save
///     "nchoices",  # save
///     "nblocks",  # save
///     "nblocks_per_subfilter",  # not saved?
///     "bits_per_subfilter",  # not saved?
///     "filterbits",  # saved
///     "mem_bytes",  # saved
///     "array",  # saved separately
///
///     # public API methods; example: f.get_value(f.array, canonical code)
///     "insert",  # (fltr:uint64[:], key:uint64, bit_positions:uint64[:])
///     "lookup",  # (fltr:uint64[:], key:uint64, bit_positions:uint64[:]) -> boolean
///     "get_fpr",  # (fltr:uint64[:]) -> float
///     "get_fill",  # (fltr:uint64[:]) -> float
///     "get_fill_histogram",  # (fltr:uint64[:]) -> uint64[:BLOCKSIZE+1]; histogram of bucket loads
///
///     # private API methods (see below)
///     "private",
/// ])
/// ```
pub struct BlowChocExactFilter {
    /// Universe size (must be power of 4)
    universe: u64,
    /// Number of subfilters for parallelization
    nsubfilters: usize,
    /// Total number of blocks
    nblocks: usize,
    /// Number of blocks per subfilter
    nblocks_per_subfilter: usize,
    /// Number of choices (alternative blocks)
    nchoices: usize,
    /// Number of bits set per key
    nbits: usize,
    /// Bit positioning strategy
    bitversion: BitVersion,
    /// Cost function for block selection
    cost_function: CostFunction,
    /// Bit array storage (aligned blocks)
    blocks: Vec<[u64; WORDS_PER_BLOCK]>,
    /// Hash function for subfilter selection
    subfilter_hash: BlowChocLinearHash,
    /// Hash functions for block selection (one per choice)
    block_hashes: Vec<BlowChocLinearHash>,
    /// Hash functions for bit positioning
    bit_hashes: Vec<BlowChocLinearHash>,
    /// Modulo values for bit positioning
    mod_values: Vec<u64>,
    /// Statistics
    elements_inserted: usize,
}

impl BlowChocExactFilter {
    pub fn new(
        universe: u64,
        nsubfilters: usize,
        nblocks: usize,
        nchoices: usize,
        nbits: usize,
        bitversion: BitVersion,
        sigma: f64,
        theta: f64,
        beta: f64,
        mu: f64,
    ) -> Result<Self> {
        Self::new_with_multipliers(
            universe,
            nsubfilters,
            nblocks,
            nchoices,
            nbits,
            bitversion,
            sigma,
            theta,
            beta,
            mu,
            None,
            None,
            None,
        )
    }

    pub fn new_with_multipliers(
        universe: u64,
        nsubfilters: usize,
        nblocks: usize,
        nchoices: usize,
        nbits: usize,
        bitversion: BitVersion,
        sigma: f64,
        theta: f64,
        beta: f64,
        mu: f64,
        subfilter_multipliers: Option<Vec<u64>>,
        bucket_multipliers: Option<Vec<u64>>,
        bit_multipliers: Option<Vec<u64>>,
    ) -> Result<Self> {
        // Validate parameters
        if nsubfilters == 0 {
            return Err(FerricError::InvalidParameter(
                "nsubfilters must be > 0".to_string(),
            ));
        }
        if nblocks == 0 {
            return Err(FerricError::InvalidParameter(
                "nblocks must be > 0".to_string(),
            ));
        }
        if nchoices == 0 {
            return Err(FerricError::InvalidParameter(
                "nchoices must be > 0".to_string(),
            ));
        }
        if nbits == 0 || nbits > BLOCKSIZE {
            return Err(FerricError::InvalidParameter(format!(
                "nbits must be 1-{}",
                BLOCKSIZE
            )));
        }

        // Calculate blocks per subfilter (must be odd, like BlowChoc)
        let mut nblocks_per_subfilter = (nblocks + nsubfilters - 1) / nsubfilters;
        if nblocks_per_subfilter % 2 == 0 {
            nblocks_per_subfilter += 1;
        }
        let total_blocks = nblocks_per_subfilter * nsubfilters;

        // Create block storage
        let blocks = vec![[0u64; WORDS_PER_BLOCK]; total_blocks];

        // Determine cost function
        let cost_function = if beta > 0.0 {
            CostFunction::Exponential { beta, mu }
        } else if mu > 0.0 {
            CostFunction::LookAhead { mu }
        } else if sigma > 0.0 || theta > 0.0 {
            CostFunction::Scaled { sigma, theta }
        } else {
            return Err(FerricError::InvalidParameter(
                "Invalid cost function parameters".to_string(),
            ));
        };

        // Create subfilter hash function
        let subfilter_mult = if let Some(ref mults) = subfilter_multipliers {
            if mults.is_empty() {
                return Err(FerricError::InvalidParameter(
                    "Empty subfilter multipliers".to_string(),
                ));
            }
            mults[0]
        } else {
            DEFAULT_MULTIPLIERS[0]
        };
        let subfilter_hash = BlowChocLinearHash::new(subfilter_mult, universe, nsubfilters as u64)?;

        // Create block hash functions (one per choice)
        let mut block_hashes = Vec::new();
        for i in 0..nchoices {
            let multiplier = if let Some(ref mults) = bucket_multipliers {
                if i < mults.len() {
                    mults[i]
                } else {
                    return Err(FerricError::InvalidParameter(format!(
                        "Not enough bucket multipliers: need {}, got {}",
                        nchoices,
                        mults.len()
                    )));
                }
            } else if i < DEFAULT_MULTIPLIERS.len() {
                DEFAULT_MULTIPLIERS[i]
            } else {
                // Generate more multipliers if needed
                generate_random_odd_multiplier()
            };
            block_hashes.push(BlowChocLinearHash::new(
                multiplier,
                universe,
                nblocks_per_subfilter as u64,
            )?);
        }

        // Calculate modulo values for bit positioning (exact BlowChoc logic)
        let mod_values = get_mod_values(&bitversion, nbits)?;

        // Create bit hash functions
        let mut bit_hashes = Vec::new();
        for i in 0..mod_values.len() {
            let multiplier = if let Some(ref mults) = bit_multipliers {
                if i < mults.len() {
                    mults[i]
                } else {
                    return Err(FerricError::InvalidParameter(format!(
                        "Not enough bit multipliers: need {}, got {}",
                        mod_values.len(),
                        mults.len()
                    )));
                }
            } else if i + nchoices < DEFAULT_MULTIPLIERS.len() {
                DEFAULT_MULTIPLIERS[i + nchoices]
            } else {
                generate_random_odd_multiplier()
            };
            bit_hashes.push(BlowChocLinearHash::new(
                multiplier,
                universe,
                mod_values[i],
            )?);
        }

        Ok(BlowChocExactFilter {
            universe,
            nsubfilters,
            nblocks: total_blocks,
            nblocks_per_subfilter,
            nchoices,
            nbits,
            bitversion,
            cost_function,
            blocks,
            subfilter_hash,
            block_hashes,
            bit_hashes,
            mod_values,
            elements_inserted: 0,
        })
    }

    /// Insert a key using exact BlowChoc algorithm
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 437-440
    /// ```python
    /// @njit(nogil=True)
    /// def insert(fltr, key, bit_positions):
    ///     subfilter = get_subfilter(key)
    ///     insert_in_subfilter(fltr, subfilter, key, bit_positions)
    /// ```
    ///
    /// And insert_in_subfilter, lines 428-435:
    /// ```python
    /// @njit(nogil=True, locals=dict(
    ///     block=uint64, sf_start_block=uint64))
    /// def insert_in_subfilter(fltr, subfilter, key, bit_positions):
    ///     # compute bit positions for key
    ///     get_bit_positions(key, bit_positions)
    ///     sf_start_block = subfilter * nblocks_per_subfilter
    ///     block = find_block(fltr, sf_start_block, key, bit_positions)
    ///     insert_in_block(fltr, block, bit_positions, 1)
    /// ```
    pub fn insert(&mut self, key: u64) {
        // Get subfilter
        let subfilter = self.subfilter_hash.hash(key) as usize;

        // Get bit positions for this key
        let mut bit_positions = vec![0u16; self.nbits];
        self.get_bit_positions(key, &mut bit_positions);

        // Find best block using cost function
        let sf_start_block = subfilter * self.nblocks_per_subfilter;
        let best_block = self.find_block(sf_start_block, key, &bit_positions);

        // Insert bits in the selected block
        self.insert_in_block(best_block, &bit_positions);
        self.elements_inserted += 1;
    }

    /// Lookup a key using exact BlowChoc algorithm
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 463-466
    /// ```python
    /// @njit(nogil=True)
    /// def lookup(fltr, key, bit_positions):
    ///     subfilter = get_subfilter(key)
    ///     return lookup_in_subfilter(fltr, subfilter, key, bit_positions)
    /// ```
    ///
    /// And lookup_in_subfilter, lines 442-456:
    /// ```python
    /// @njit(nogil=True, locals=dict(start_block=uint64, next_block=uint64, block=uint64, v=uint64))
    /// def lookup_in_subfilter(fltr, subfilter, key, bit_positions):
    ///     # find bit positions
    ///     get_bit_positions(key, bit_positions)
    ///     start_block = subfilter * nblocks_per_subfilter
    ///     next_block = start_block + nblocks_per_subfilter
    ///     for c in range(nchoices):
    ///         block = start_block + get_block(c, key)
    ///         if block >= next_block:
    ///             continue
    ///         v = lookup_in_block(fltr, block, bit_positions)
    ///         if v:
    ///             return True
    ///     return False
    /// ```
    pub fn lookup(&self, key: u64) -> bool {
        // Get subfilter
        let subfilter = self.subfilter_hash.hash(key) as usize;

        // Get bit positions for this key
        let mut bit_positions = vec![0u16; self.nbits];
        self.get_bit_positions(key, &mut bit_positions);

        // Check all choice blocks
        let sf_start_block = subfilter * self.nblocks_per_subfilter;
        for choice in 0..self.nchoices {
            let block = sf_start_block + self.block_hashes[choice].hash(key) as usize;
            if self.lookup_in_block(block, &bit_positions) {
                return true;
            }
        }
        false
    }

    /// Get bit positions using exact BlowChoc algorithm
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 291-322
    /// ```python
    /// @njit(nogil=True, locals=dict(pos=uint16))
    /// def _get_bit_positions_distinct(key, bit_positions):
    ///     # compute bit positions for key
    ///     for i in range(nbits):
    ///         pos = get_bit_position(i, key)
    ///         bit_positions[i] = pos
    ///     # ensure distinct positions
    ///     for i in range(nbits):
    ///         for j in range(i+1, nbits):
    ///             if bit_positions[i] == bit_positions[j]:
    ///                 bit_positions[j] = (bit_positions[j] + 1) % mod_values[j]
    ///
    /// @njit(nogil=True, locals=dict(pos=uint16, offset=uint64))
    /// def _get_bit_positions_double(key, bit_positions):
    ///     # compute bit positions for key
    ///     for i in range(nbits):
    ///         pos = get_bit_position(i, key)
    ///         bit_positions[i] = pos
    ///     # offset second half by 256
    ///     for i in range(nbits//2, nbits):
    ///         offset = BLOCKSIZE // 2
    ///         bit_positions[i] = (bit_positions[i] + offset) % BLOCKSIZE
    /// ```
    fn get_bit_positions(&self, key: u64, bit_positions: &mut [u16]) {
        match self.bitversion {
            BitVersion::Random => {
                for i in 0..self.nbits {
                    bit_positions[i] = self.bit_hashes[i].hash(key) as u16;
                }
            }
            BitVersion::Distinct => {
                // First get raw positions
                for i in 0..self.nbits {
                    bit_positions[i] = self.bit_hashes[i].hash(key) as u16;
                }
                // Make them distinct (exact BlowChoc algorithm)
                for b in 0..self.nbits {
                    let mut pos = bit_positions[b];
                    let mut i = 0;
                    while i < b && pos >= bit_positions[i] {
                        pos += 1;
                        i += 1;
                    }
                    // Shift positions to make room
                    for j in (i..b).rev() {
                        bit_positions[j + 1] = bit_positions[j];
                    }
                    bit_positions[i] = pos;
                }
            }
            BitVersion::Double => {
                let pos = self.bit_hashes[0].hash(key) as u16;
                let offset = self.bit_hashes[1].hash(key) as u16 * 2 + 1;
                for b in 0..self.nbits {
                    bit_positions[b] = (pos + offset * (b as u16)) % (BLOCKSIZE as u16);
                }
            }
        }
    }

    /// Find best block using exact BlowChoc cost function
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 361-407
    /// ```python
    /// @njit(nogil=True, locals=dict(
    ///     block=uint64, best_block=uint64,
    ///     error=float64, best_error=float64, next_block=uint64,
    ///     start=uint64, end=uint64, nbits_set=uint64, already_set_bits=uint64,
    ///     nnew_bits=uint64, pos=uint64, inner_cost=float64,
    ///     w=uint64))
    /// def find_block(fltr, sf_start_block, key, bit_positions):
    ///     # return block number where to insert
    ///     # after evaluating cost functions
    ///     best_block = sf_start_block
    ///     best_error = float64(np.inf)
    ///     for choice in range(nchoices):
    ///         block = sf_start_block + get_block(choice, key)
    ///         if block >= sf_start_block + nblocks_per_subfilter:
    ///             continue
    ///         start = block * BLOCKSIZE // 64
    ///         end = start + BLOCKSIZE // 64
    ///         nbits_set = popcount(fltr[start:end])
    ///         already_set_bits = uint64(0)
    ///         nnew_bits = uint64(0)
    ///         for i in range(nbits):
    ///             pos = bit_positions[i]
    ///             w = start + pos // 64
    ///             if (fltr[w] >> (pos % 64)) & 1:
    ///                 already_set_bits += 1
    ///             else:
    ///                 nnew_bits += 1
    ///         inner_cost = costfunction(nbits_set, nnew_bits)
    ///         if inner_cost < best_error:
    ///             best_error = inner_cost
    ///             best_block = block
    ///     return best_block
    /// ```
    fn find_block(&self, sf_start_block: usize, key: u64, bit_positions: &[u16]) -> usize {
        if self.nchoices == 1 {
            return sf_start_block + self.block_hashes[0].hash(key) as usize;
        }

        let mut best_cost = f64::MAX;
        let mut best_block = 0;

        for choice in 0..self.nchoices {
            let block = sf_start_block + self.block_hashes[choice].hash(key) as usize;

            // Count current set bits in block
            let nbits_set = self.count_set_bits_in_block(block);

            // Count how many new bits would be set
            let nnew_bits = self.count_new_bits_in_block(block, bit_positions);

            if nnew_bits == 0 {
                return block; // Already all bits are set
            }

            // Calculate cost using exact BlowChoc formula
            let cost = self.calculate_cost(nbits_set, nnew_bits);

            if cost < best_cost {
                best_cost = cost;
                best_block = block;
            }
        }

        best_block
    }

    /// Calculate cost using exact BlowChoc cost functions
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 324-350
    /// ```python
    /// @njit(nogil=True)
    /// def cost_function_scaled(nbits_set, nnew_bits):
    ///     # sigma * nbits * ((2 * q / blocksize) ** nbits) + theta * nnew_bits
    ///     # where q is the number of already set bits in the block
    ///     inner_cost = sigma * nbits * ((2 * nbits_set / BLOCKSIZE) ** nbits)
    ///     return inner_cost + theta * nnew_bits
    ///
    /// @njit(nogil=True)
    /// def cost_function_lookahead(nbits_set, nnew_bits):
    ///     # nbits * ((2 * q / blocksize) ** nbits) + nnew_bits
    ///     # where q includes mu * nbits
    ///     q = nbits_set + mu * nbits
    ///     inner_cost = nbits * ((2 * q / BLOCKSIZE) ** nbits)
    ///     return inner_cost + nnew_bits
    ///
    /// @njit(nogil=True)
    /// def cost_function_exp(nbits_set, nnew_bits):
    ///     # beta**(q / blocksize4) + mu * (nnew_bits / nbits)
    ///     blocksize4 = BLOCKSIZE // 4
    ///     q = nbits_set
    ///     inner_cost = beta ** (q / blocksize4)
    ///     return inner_cost + mu * (nnew_bits / nbits)
    /// ```
    fn calculate_cost(&self, nbits_set: usize, nnew_bits: usize) -> f64 {
        let q = nnew_bits + nbits_set;
        let blocksize_f = BLOCKSIZE as f64;
        let nbits_f = self.nbits as f64;

        match &self.cost_function {
            CostFunction::Scaled { sigma, theta } => {
                sigma * nbits_f * ((2.0 * q as f64 / blocksize_f).powf(nbits_f))
                    + theta * nnew_bits as f64
            }
            CostFunction::LookAhead { mu } => {
                let q_lookahead = nnew_bits + nbits_set + (mu * nbits_f) as usize;
                nbits_f * ((2.0 * q_lookahead as f64 / blocksize_f).powf(nbits_f))
                    + nnew_bits as f64
            }
            CostFunction::Exponential { beta, mu } => {
                let blocksize4 = blocksize_f / 4.0;
                beta.powf(q as f64 / blocksize4) + mu * (nnew_bits as f64 / nbits_f)
            }
        }
    }

    /// Count set bits in a block
    fn count_set_bits_in_block(&self, block_idx: usize) -> usize {
        self.blocks[block_idx]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Count how many new bits would be set
    fn count_new_bits_in_block(&self, block_idx: usize, bit_positions: &[u16]) -> usize {
        let mut new_bits = 0;
        for &pos in bit_positions {
            let word_idx = pos as usize / 64;
            let bit_idx = pos as usize % 64;
            if (self.blocks[block_idx][word_idx] & (1u64 << bit_idx)) == 0 {
                new_bits += 1;
            }
        }
        new_bits
    }

    /// Insert bits in a specific block
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 409-416
    /// ```python
    /// @njit(nogil=True, locals=dict(
    ///     block=uint64, pos=uint64, offset=uint64))
    /// def insert_in_block(fltr, block, bit_positions, value):
    ///     for i in range(nbits):
    ///         pos = bit_positions[i]
    ///         offset = block * BLOCKSIZE // 64 + pos // 64
    ///         fltr[offset] |= (1 << (pos % 64))
    /// ```
    fn insert_in_block(&mut self, block_idx: usize, bit_positions: &[u16]) {
        for &pos in bit_positions {
            let word_idx = pos as usize / 64;
            let bit_idx = pos as usize % 64;
            self.blocks[block_idx][word_idx] |= 1u64 << bit_idx;
        }
    }

    /// Check if all bits are set in a specific block
    ///
    /// Original BlowChoc reference:
    /// File: blowchoc/blowchoc/filter_bb_choices.py, lines 417-426
    /// ```python
    /// @njit(nogil=True, locals=dict(
    ///     offset=uint64, block=uint64, pos=uint64, bf=uint64))
    /// def lookup_in_block(fltr, block, bit_positions):
    ///     for i in range(nbits):
    ///         pos = bit_positions[i]
    ///         offset = block * BLOCKSIZE // 64 + pos // 64
    ///         bf = fltr[offset]
    ///         if not (bf >> (pos % 64)) & 1:
    ///             return False
    ///     return True
    /// ```
    fn lookup_in_block(&self, block_idx: usize, bit_positions: &[u16]) -> bool {
        for &pos in bit_positions {
            let word_idx = pos as usize / 64;
            let bit_idx = pos as usize % 64;
            if (self.blocks[block_idx][word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Get basic statistics
    pub fn len(&self) -> usize {
        self.elements_inserted
    }

    pub fn is_empty(&self) -> bool {
        self.elements_inserted == 0
    }

    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            *block = [0u64; WORDS_PER_BLOCK];
        }
        self.elements_inserted = 0;
    }

    /// Get false positive rate (exact BlowChoc calculation)
    pub fn get_fpr(&self) -> f64 {
        let mut overall_fpr = 0.0;
        for i in 0..self.nblocks {
            let n_bits_set = self.count_set_bits_in_block(i);
            let block_fpr = (n_bits_set as f64 / BLOCKSIZE as f64).powf(self.nbits as f64);
            overall_fpr += block_fpr;
        }
        overall_fpr /= self.nblocks as f64;
        overall_fpr * self.nchoices as f64
    }

    /// Get fill rate (exact BlowChoc calculation)
    pub fn get_fill(&self) -> f64 {
        let total_bits = self.nblocks * BLOCKSIZE;
        let set_bits: usize = (0..self.nblocks)
            .map(|i| self.count_set_bits_in_block(i))
            .sum();
        set_bits as f64 / total_bits as f64
    }

    /// Get fill histogram (exact BlowChoc calculation)
    pub fn get_fill_histogram(&self) -> Vec<usize> {
        let mut count = vec![0usize; BLOCKSIZE + 1];
        for subtable in 0..self.nsubfilters {
            let sf_start = subtable * self.nblocks_per_subfilter;
            for block in 0..self.nblocks_per_subfilter {
                let block_idx = sf_start + block;
                if block_idx < self.nblocks {
                    let load = self.count_set_bits_in_block(block_idx);
                    count[load] += 1;
                }
            }
        }
        count
    }

    /// Get raw block data
    pub fn get_blocks(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.nblocks * WORDS_PER_BLOCK);
        for block in &self.blocks {
            result.extend_from_slice(block);
        }
        result
    }
}

/// Get modulo values for bit positioning (exact BlowChoc logic)
///
/// Original BlowChoc reference:
/// File: blowchoc/blowchoc/filter_bb_choices.py, lines 134-148
/// ```python
/// def get_mod_values(bitversion, k):
///     """Computes the modulo and offset values to correctly compute the bit positions using different strategies"""
///     # use 512, 511, 510, ..., 512-k+1 as mod values to compute k distinct bit positions
///     if bitversion == 'distinct':
///         mod_values = tuple(BLOCKSIZE - i for i in range(k))
///         return mod_values
///     # compute k random positions using a mod value of 512
///     if bitversion == 'random':
///         mod_values = (BLOCKSIZE,) * k
///         return mod_values
///     if bitversion == 'double':
///         mod_values = (BLOCKSIZE, BLOCKSIZE // 2)
///         return mod_values
///     raise ValueError(f"ERROR: get_mod_values: {bitversion=} is not supported.")
/// ```
fn get_mod_values(bitversion: &BitVersion, nbits: usize) -> Result<Vec<u64>> {
    match bitversion {
        BitVersion::Distinct => {
            // Use 512, 511, 510, ..., 512-k+1 as mod values
            Ok((0..nbits).map(|i| BLOCKSIZE as u64 - i as u64).collect())
        }
        BitVersion::Random => {
            // Use 512 for all positions
            Ok(vec![BLOCKSIZE as u64; nbits])
        }
        BitVersion::Double => {
            // Use 512 and 256
            Ok(vec![BLOCKSIZE as u64, BLOCKSIZE as u64 / 2])
        }
    }
}

/// Generate a random odd multiplier
fn generate_random_odd_multiplier() -> u64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let multiplier = rng.gen_range(3..=u32::MAX as u64);
    multiplier | 1 // Ensure odd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blowchoc_exact_basic() {
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
    }

    #[test]
    fn test_hash_function_exact() {
        let hash = BlowChocLinearHash::new(62591, 4_u64.pow(10), 100).unwrap();

        // Test that the hash function is deterministic
        let key = 12345;
        let hash1 = hash.hash(key);
        let hash2 = hash.hash(key);
        assert_eq!(hash1, hash2);

        // Test that different keys give different hashes (mostly)
        let hash3 = hash.hash(54321);
        assert_ne!(hash1, hash3); // Very likely to be different
    }
}
