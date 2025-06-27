//! Basic usage examples for ferric-bloom

use ferric_bloom::blocked_bloom::CostFunction;
use ferric_bloom::{BlockedBloomFilter, BloomFilter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ferric Bloom Filter Examples ===\n");

    // Example 1: Basic Bloom Filter
    println!("1. Basic Bloom Filter:");
    let mut bloom = BloomFilter::new(1000, 7)?;

    // Insert some data
    let test_data = [42u64, 1337, 9999, 12345, 67890];
    for &item in &test_data {
        bloom.insert(item);
    }

    // Check membership
    for &item in &test_data {
        println!("  {} in filter: {}", item, bloom.contains(item));
    }

    // Check some items not inserted
    for &item in &[1u64, 2, 3, 4, 5] {
        println!("  {} in filter: {}", item, bloom.contains(item));
    }

    println!("  {}", bloom.stats());
    println!();

    // Example 2: Blocked Bloom Filter
    println!("2. Blocked Bloom Filter:");
    let mut blocked = BlockedBloomFilter::new(20, 8, 2)?;

    // Try different cost functions
    blocked.set_cost_function(CostFunction::MinFpr {
        sigma: 1.0,
        theta: 1.0,
    });

    // Insert test data
    for &item in &test_data {
        blocked.insert(item);
    }

    // Check membership
    for &item in &test_data {
        println!("  {} in blocked filter: {}", item, blocked.contains(item));
    }

    println!("  {}", blocked.stats());
    println!();

    // Example 3: Performance comparison
    println!("3. Performance comparison:");
    let num_items = 1_000; // Reduced from 10,000 for faster demo

    // Time standard Bloom filter
    let start = std::time::Instant::now();
    let mut bloom_large = BloomFilter::new(num_items, 10)?;
    for i in 0..num_items {
        bloom_large.insert(i as u64);
    }
    let bloom_insert_time = start.elapsed();

    let start = std::time::Instant::now();
    let mut found = 0;
    for i in 0..num_items {
        if bloom_large.contains(i as u64) {
            found += 1;
        }
    }
    let bloom_query_time = start.elapsed();

    // Time blocked Bloom filter - use more blocks for better performance
    let start = std::time::Instant::now();
    let mut blocked_large = BlockedBloomFilter::new(200, 8, 2)?; // Increased blocks from 100 to 200
    for i in 0..num_items {
        blocked_large.insert(i as u64);
    }
    let blocked_insert_time = start.elapsed();

    let start = std::time::Instant::now();
    let mut blocked_found = 0;
    for i in 0..num_items {
        if blocked_large.contains(i as u64) {
            blocked_found += 1;
        }
    }
    let blocked_query_time = start.elapsed();

    println!("  Standard Bloom:");
    println!(
        "    Insert: {:?} ({:.2} M ops/sec)",
        bloom_insert_time,
        num_items as f64 / bloom_insert_time.as_secs_f64() / 1_000_000.0
    );
    println!(
        "    Query:  {:?} ({:.2} M ops/sec)",
        bloom_query_time,
        num_items as f64 / bloom_query_time.as_secs_f64() / 1_000_000.0
    );
    println!("    Found: {}/{}", found, num_items);

    println!("  Blocked Bloom:");
    println!(
        "    Insert: {:?} ({:.2} M ops/sec)",
        blocked_insert_time,
        num_items as f64 / blocked_insert_time.as_secs_f64() / 1_000_000.0
    );
    println!(
        "    Query:  {:?} ({:.2} M ops/sec)",
        blocked_query_time,
        num_items as f64 / blocked_query_time.as_secs_f64() / 1_000_000.0
    );
    println!("    Found: {}/{}", blocked_found, num_items);

    // Example 4: DNA k-mer usage
    println!("\n4. DNA k-mer example:");
    let mut kmer_filter = BlockedBloomFilter::new(100, 8, 2)?;

    // Simple DNA k-mer encoding function
    fn encode_kmer(kmer: &str) -> u64 {
        kmer.bytes().fold(0u64, |acc, base| {
            let code = match base {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 0,
            };
            (acc << 2) | code
        })
    }

    // Insert k-mers from a sequence
    let dna_sequence = "ATCGATCGATCGAAAGGGCCCTTTAAATTTCCCGGG";
    let kmer_size = 6;

    println!("  DNA sequence: {}", dna_sequence);
    println!("  Inserting {}-mers:", kmer_size);

    for i in 0..=(dna_sequence.len().saturating_sub(kmer_size)) {
        let kmer = &dna_sequence[i..i + kmer_size];
        let encoded = encode_kmer(kmer);
        kmer_filter.insert(encoded);
        println!("    {} -> {}", kmer, encoded);
    }

    // Query for specific k-mers
    let query_kmers = ["ATCGAT", "GGCCCT", "TTTAAA", "XXXXXX"];
    println!("  Querying k-mers:");
    for kmer in &query_kmers {
        let encoded = encode_kmer(kmer);
        let present = kmer_filter.contains(encoded);
        println!(
            "    {} -> {} ({})",
            kmer,
            encoded,
            if present { "FOUND" } else { "NOT FOUND" }
        );
    }

    println!("  {}", kmer_filter.stats());

    Ok(())
}
