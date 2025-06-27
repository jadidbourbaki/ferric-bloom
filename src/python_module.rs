//! Python bindings for ferric-bloom using PyO3

use crate::{
    blocked_bloom::CostFunction as RustCostFunction,
    hash::create_hash_functions,
    utils::{optimal_blocked_bloom_parameters, optimal_bloom_parameters},
    BlockedBloomFilter as RustBlockedBloomFilter, BloomFilter as RustBloomFilter,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Python wrapper for BloomFilter
#[pyclass(name = "BloomFilter")]
struct PyBloomFilter {
    inner: RustBloomFilter,
}

#[pymethods]
impl PyBloomFilter {
    #[new]
    fn new(capacity_or_num_bits: usize, num_hash_functions: usize) -> PyResult<Self> {
        let filter = RustBloomFilter::new(capacity_or_num_bits, num_hash_functions)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyBloomFilter { inner: filter })
    }

    #[classmethod]
    fn with_size(_cls: &PyType, num_bits: usize, num_hash_functions: usize) -> PyResult<Self> {
        let filter = RustBloomFilter::with_size(num_bits, num_hash_functions)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyBloomFilter { inner: filter })
    }

    fn insert(&mut self, key: u64) -> bool {
        self.inner.insert(key);
        true
    }

    fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn estimated_fpr(&self) -> f64 {
        self.inner.estimated_fpr()
    }

    fn load_factor(&self) -> f64 {
        self.inner.load_factor()
    }

    fn stats(&self) -> String {
        self.inner.stats().to_string()
    }

    fn __str__(&self) -> String {
        self.stats()
    }

    fn __repr__(&self) -> String {
        format!(
            "BloomFilter(capacity={}, hash_functions={})",
            self.inner.capacity(),
            self.inner.num_hash_functions()
        )
    }
}

/// Python wrapper for CostFunction
#[pyclass(name = "CostFunction")]
#[derive(Clone)]
struct PyCostFunction {
    inner: RustCostFunction,
}

#[pymethods]
impl PyCostFunction {
    #[classmethod]
    fn min_load(_cls: &PyType) -> Self {
        PyCostFunction {
            inner: RustCostFunction::MinLoad,
        }
    }

    #[classmethod]
    fn min_fpr(_cls: &PyType, sigma: f64, theta: f64) -> Self {
        PyCostFunction {
            inner: RustCostFunction::MinFpr { sigma, theta },
        }
    }

    #[classmethod]
    fn look_ahead(_cls: &PyType, mu: f64) -> Self {
        PyCostFunction {
            inner: RustCostFunction::LookAhead { mu },
        }
    }

    fn __str__(&self) -> String {
        match &self.inner {
            RustCostFunction::MinLoad => "MinLoad".to_string(),
            RustCostFunction::MinFpr { sigma, theta } => {
                format!("MinFpr(sigma={}, theta={})", sigma, theta)
            }
            RustCostFunction::LookAhead { mu } => {
                format!("LookAhead(mu={})", mu)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("CostFunction.{}", self.__str__())
    }
}

/// Python wrapper for BlockedBloomFilter
#[pyclass(name = "BlockedBloomFilter")]
struct PyBlockedBloomFilter {
    inner: RustBlockedBloomFilter,
}

#[pymethods]
impl PyBlockedBloomFilter {
    #[new]
    fn new(num_blocks: usize, bits_per_key: usize, num_choices: usize) -> PyResult<Self> {
        let filter = RustBlockedBloomFilter::new(num_blocks, bits_per_key, num_choices)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyBlockedBloomFilter { inner: filter })
    }

    fn insert(&mut self, key: u64) -> bool {
        self.inner.insert(key);
        true
    }

    fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn num_blocks(&self) -> usize {
        self.inner.num_blocks()
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn estimated_fpr(&self) -> f64 {
        self.inner.estimated_fpr()
    }

    fn load_factor(&self) -> f64 {
        self.inner.load_factor()
    }

    fn set_cost_function(&mut self, cost_fn: &PyCostFunction) {
        self.inner.set_cost_function(cost_fn.inner.clone());
    }

    fn stats(&self) -> String {
        self.inner.stats().to_string()
    }

    /// Get the raw block data as a numpy array
    fn get_blocks<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        let blocks = self.inner.get_blocks();
        blocks.into_pyarray(py)
    }

    /// Get block load distribution
    fn get_block_loads<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        let loads = self.inner.get_block_loads();
        loads.into_pyarray(py)
    }

    fn __str__(&self) -> String {
        self.stats()
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockedBloomFilter(blocks={}, bits_per_key={}, choices={})",
            self.inner.num_blocks(),
            self.inner.bits_per_key(),
            self.inner.num_choices()
        )
    }
}

/// Optimal Bloom filter parameters
#[pyfunction]
fn py_optimal_bloom_parameters(
    expected_elements: usize,
    desired_fpr: f64,
    memory_limit_bits: Option<usize>,
) -> (usize, usize, f64) {
    let params = optimal_bloom_parameters(expected_elements, desired_fpr, memory_limit_bits);
    (
        params.optimal_num_bits,
        params.optimal_num_hashes,
        params.expected_fpr,
    )
}

/// Optimal Blocked Bloom filter parameters
#[pyfunction]
fn py_optimal_blocked_bloom_parameters(
    expected_elements: usize,
    desired_fpr: f64,
    num_choices: usize,
) -> (usize, usize) {
    optimal_blocked_bloom_parameters(expected_elements, desired_fpr, num_choices)
}

/// Create a list of diverse hash function multipliers
#[pyfunction]
fn py_create_hash_multipliers(count: usize) -> Vec<u64> {
    let hash_fns = create_hash_functions(count);
    hash_fns
        .into_iter()
        .filter_map(|hf| {
            // Try to downcast to LinearHash to get multiplier
            if let Ok(linear) = hf.name().parse::<u64>() {
                Some(linear)
            } else {
                None
            }
        })
        .collect()
}

/// DNA k-mer encoding utility
#[pyfunction]
fn encode_dna_kmer(kmer: &str) -> u64 {
    kmer.bytes().fold(0u64, |acc, base| {
        let code = match base.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0, // Default to A for invalid characters
        };
        (acc << 2) | code
    })
}

/// Decode a DNA k-mer from u64 encoding
#[pyfunction]
fn decode_dna_kmer(encoded: u64, kmer_length: usize) -> String {
    let mut result = String::with_capacity(kmer_length);
    let bases = [b'A', b'C', b'G', b'T'];

    for i in (0..kmer_length).rev() {
        let shift = i * 2;
        let base_code = ((encoded >> shift) & 3) as usize;
        result.push(bases[base_code] as char);
    }

    result
}

/// Performance testing utilities
#[pyfunction]
fn benchmark_bloom_filter(
    capacity: usize,
    num_hash_functions: usize,
    num_operations: usize,
) -> PyResult<(f64, f64)> {
    use std::time::Instant;

    let mut filter = RustBloomFilter::new(capacity, num_hash_functions)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Benchmark insertions
    let start = Instant::now();
    for i in 0..num_operations {
        filter.insert(i as u64);
    }
    let insert_time = start.elapsed().as_secs_f64();

    // Benchmark queries
    let start = Instant::now();
    for i in 0..num_operations {
        filter.contains(i as u64);
    }
    let query_time = start.elapsed().as_secs_f64();

    Ok((insert_time, query_time))
}

#[pyfunction]
fn benchmark_blocked_bloom_filter(
    num_blocks: usize,
    bits_per_key: usize,
    num_choices: usize,
    num_operations: usize,
) -> PyResult<(f64, f64)> {
    use std::time::Instant;

    let mut filter = RustBlockedBloomFilter::new(num_blocks, bits_per_key, num_choices)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Benchmark insertions
    let start = Instant::now();
    for i in 0..num_operations {
        filter.insert(i as u64);
    }
    let insert_time = start.elapsed().as_secs_f64();

    // Benchmark queries
    let start = Instant::now();
    for i in 0..num_operations {
        filter.contains(i as u64);
    }
    let query_time = start.elapsed().as_secs_f64();

    Ok((insert_time, query_time))
}

/// Python module definition
#[pymodule]
fn ferric_bloom(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBloomFilter>()?;
    m.add_class::<PyBlockedBloomFilter>()?;
    m.add_class::<PyCostFunction>()?;

    m.add_function(wrap_pyfunction!(py_optimal_bloom_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimal_blocked_bloom_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(py_create_hash_multipliers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_dna_kmer, m)?)?;
    m.add_function(wrap_pyfunction!(decode_dna_kmer, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_bloom_filter, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_blocked_bloom_filter, m)?)?;

    // Add module constants
    m.add("BLOCK_SIZE", 512)?;
    m.add("__version__", "0.1.0")?;

    Ok(())
}

fn main() {
    // This binary is only used when building Python extension
    println!(
        "Use 'maturin develop' or 'pip install maturin && maturin build' to build Python extension"
    );
}
