//! Python bindings for ferric-bloom using PyO3

use crate::blowchoc::{BitVersion, BlowChocExactFilter};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// Python wrapper for BlowChocExactFilter
#[pyclass(name = "BlowChocExactFilter")]
struct PyBlowChocExactFilter {
    inner: BlowChocExactFilter,
}

#[pymethods]
impl PyBlowChocExactFilter {
    #[new]
    fn new(
        universe: u64,
        nsubfilters: usize,
        nblocks: usize,
        nchoices: usize,
        nbits: usize,
        bitversion: String,
        sigma: f64,
        theta: f64,
        beta: f64,
        mu: f64,
    ) -> PyResult<Self> {
        let bit_version = BitVersion::from_str(&bitversion)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let filter = BlowChocExactFilter::new(
            universe,
            nsubfilters,
            nblocks,
            nchoices,
            nbits,
            bit_version,
            sigma,
            theta,
            beta,
            mu,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(PyBlowChocExactFilter { inner: filter })
    }

    fn insert(&mut self, key: u64) -> bool {
        self.inner.insert(key);
        true
    }

    fn lookup(&self, key: u64) -> bool {
        self.inner.lookup(key)
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

    fn get_fpr(&self) -> f64 {
        self.inner.get_fpr()
    }

    fn get_fill(&self) -> f64 {
        self.inner.get_fill()
    }

    fn get_fill_histogram<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        let histogram = self.inner.get_fill_histogram();
        histogram.into_pyarray(py)
    }

    fn backend(&self) -> &str {
        "rust"
    }

    fn filtertype(&self) -> &str {
        "bb_choices"
    }

    fn array<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        let blocks = self.inner.get_blocks();
        blocks.into_pyarray(py)
    }

    fn filterbits(&self) -> usize {
        self.inner.get_blocks().len() * 64
    }

    fn mem_bytes(&self) -> usize {
        self.inner.get_blocks().len() * 8
    }

    // Alias for insert (BlowChoc compatibility)
    fn add(&mut self, key: u64) -> bool {
        self.insert(key)
    }

    fn stats(&self) -> String {
        format!(
            "BlowChocExactFilter Statistics:\n\
            Elements: {}\n\
            FPR: {:.6}\n\
            Fill: {:.6}\n\
            Memory: {} bytes\n\
            Filter bits: {}",
            self.inner.len(),
            self.inner.get_fpr(),
            self.inner.get_fill(),
            self.mem_bytes(),
            self.filterbits()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "BlowChocExactFilter(elements={}, fpr={:.6}, fill={:.6})",
            self.inner.len(),
            self.inner.get_fpr(),
            self.inner.get_fill()
        )
    }

    fn __repr__(&self) -> String {
        format!("BlowChocExactFilter(len={})", self.inner.len())
    }
}

/// Python wrapper for BlowChocExactFilterWithMultipliers
#[pyclass(name = "BlowChocExactFilterWithMultipliers")]
struct PyBlowChocExactFilterWithMultipliers {
    inner: BlowChocExactFilter,
}

#[pymethods]
impl PyBlowChocExactFilterWithMultipliers {
    #[new]
    fn new(
        universe: u64,
        nsubfilters: usize,
        nblocks: usize,
        nchoices: usize,
        nbits: usize,
        bitversion: String,
        sigma: f64,
        theta: f64,
        beta: f64,
        mu: f64,
        subfilter_multipliers: Vec<u64>,
        bucket_multipliers: Vec<u64>,
        bit_multipliers: Vec<u64>,
    ) -> PyResult<Self> {
        let bit_version = BitVersion::from_str(&bitversion)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let filter = BlowChocExactFilter::new_with_multipliers(
            universe,
            nsubfilters,
            nblocks,
            nchoices,
            nbits,
            bit_version,
            sigma,
            theta,
            beta,
            mu,
            Some(subfilter_multipliers),
            Some(bucket_multipliers),
            Some(bit_multipliers),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(PyBlowChocExactFilterWithMultipliers { inner: filter })
    }

    fn insert(&mut self, key: u64) -> bool {
        self.inner.insert(key);
        true
    }

    fn lookup(&self, key: u64) -> bool {
        self.inner.lookup(key)
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

    fn get_fpr(&self) -> f64 {
        self.inner.get_fpr()
    }

    fn get_fill(&self) -> f64 {
        self.inner.get_fill()
    }

    fn get_fill_histogram<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        let histogram = self.inner.get_fill_histogram();
        histogram.into_pyarray(py)
    }

    fn backend(&self) -> &str {
        "rust"
    }

    fn filtertype(&self) -> &str {
        "bb_choices"
    }

    fn array<'py>(&self, py: Python<'py>) -> &'py PyArray1<u64> {
        let blocks = self.inner.get_blocks();
        blocks.into_pyarray(py)
    }

    fn filterbits(&self) -> usize {
        self.inner.get_blocks().len() * 64
    }

    fn mem_bytes(&self) -> usize {
        self.inner.get_blocks().len() * 8
    }

    // Alias for insert (BlowChoc compatibility)
    fn add(&mut self, key: u64) -> bool {
        self.insert(key)
    }

    fn stats(&self) -> String {
        format!(
            "BlowChocExactFilterWithMultipliers Statistics:\n\
            Elements: {}\n\
            FPR: {:.6}\n\
            Fill: {:.6}\n\
            Memory: {} bytes\n\
            Filter bits: {}",
            self.inner.len(),
            self.inner.get_fpr(),
            self.inner.get_fill(),
            self.mem_bytes(),
            self.filterbits()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "BlowChocExactFilterWithMultipliers(elements={}, fpr={:.6}, fill={:.6})",
            self.inner.len(),
            self.inner.get_fpr(),
            self.inner.get_fill()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "BlowChocExactFilterWithMultipliers(len={})",
            self.inner.len()
        )
    }
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

/// Python module definition
#[pymodule]
fn ferric_bloom(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBlowChocExactFilter>()?;
    m.add_class::<PyBlowChocExactFilterWithMultipliers>()?;

    m.add_function(wrap_pyfunction!(encode_dna_kmer, m)?)?;
    m.add_function(wrap_pyfunction!(decode_dna_kmer, m)?)?;

    // Add module constants
    m.add("BLOCK_SIZE", 512)?;
    m.add("__version__", "0.1.0")?;

    Ok(())
}
