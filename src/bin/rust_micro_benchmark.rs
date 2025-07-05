use ferric_bloom::{BitVersion, BlowChocExactFilter};
use std::time::Instant;

fn main() {
    println!("🦀 Rust Micro Benchmark - Native Performance Test");
    println!("{}", "=".repeat(55));

    // Test range: 0 to 200 in steps of 25 (same as Python benchmark)
    let element_counts: Vec<usize> = (0..=200).step_by(25).collect();

    println!("Testing element counts: {:?}", element_counts);
    println!();

    // Results storage
    let mut results = Vec::new();

    for &n_elements in &element_counts {
        println!("🔬 Testing {} elements...", n_elements);

        if n_elements == 0 {
            // Special case for 0 elements
            results.push((n_elements, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0));
            continue;
        }

        // Configuration (same as Python benchmark)
        let universe = 4_u64.pow(12); // 4^12 = 16,777,216
        let nblocks = 100;
        let nbits = 8;
        let nchoices = 2;

        // Create filter
        let start = Instant::now();
        let mut filter = BlowChocExactFilter::new(
            universe,
            1, // nsubfilters
            nblocks,
            nchoices,
            nbits,
            BitVersion::Random,
            0.0, // sigma
            1.0, // theta
            0.0, // beta
            0.0, // mu
        )
        .unwrap();
        let creation_time = start.elapsed().as_secs_f64();

        // Generate test data
        let test_keys: Vec<u64> = (0..n_elements as u64).collect();
        let query_keys: Vec<u64> =
            (n_elements as u64..(n_elements + 100.min(n_elements)) as u64).collect();

        // Insert benchmark
        let start = Instant::now();
        for &key in &test_keys {
            filter.insert(key);
        }
        let insert_time = start.elapsed().as_secs_f64();

        // Query benchmark (successful lookups)
        let sample_keys: Vec<u64> = test_keys
            .iter()
            .take(10.min(test_keys.len()))
            .copied()
            .collect();
        let start = Instant::now();
        let mut hits = 0;
        for &key in &sample_keys {
            if filter.lookup(key) {
                hits += 1;
            }
        }
        let _successful_query_time = start.elapsed().as_secs_f64();

        // Query benchmark (false positive test)
        let start = Instant::now();
        let mut false_positives = 0;
        for &key in &query_keys {
            if filter.lookup(key) {
                false_positives += 1;
            }
        }
        let fp_query_time = start.elapsed().as_secs_f64();

        // Calculate rates
        let insert_rate = if insert_time > 0.0 {
            n_elements as f64 / insert_time
        } else {
            f64::INFINITY
        };
        let query_rate = if fp_query_time > 0.0 {
            query_keys.len() as f64 / fp_query_time
        } else {
            f64::INFINITY
        };
        let false_positive_rate = if !query_keys.is_empty() {
            false_positives as f64 / query_keys.len() as f64
        } else {
            0.0
        };

        results.push((
            n_elements,
            creation_time,
            insert_time,
            fp_query_time,
            insert_rate,
            query_rate,
            hits,
            false_positives,
            false_positive_rate,
        ));

        println!(
            "   ✅ Done - Insert rate: {:.0} ops/s, Query rate: {:.0} ops/s",
            insert_rate, query_rate
        );
    }

    // Print results in CSV format for easy comparison with Python benchmark
    println!("\n📊 Results (CSV format):");
    println!("elements,creation_time,insert_time,query_time,insert_rate,query_rate,hits,false_positives,false_positive_rate");

    for (
        elements,
        creation_time,
        insert_time,
        query_time,
        insert_rate,
        query_rate,
        hits,
        false_positives,
        fpr,
    ) in &results
    {
        println!(
            "{},{:.6},{:.6},{:.6},{:.0},{:.0},{},{},{:.6}",
            elements,
            creation_time,
            insert_time,
            query_time,
            insert_rate,
            query_rate,
            hits,
            false_positives,
            fpr
        );
    }

    // Print summary comparison
    if let Some(&(
        max_elements,
        creation_time,
        _insert_time,
        _query_time,
        insert_rate,
        query_rate,
        _hits,
        _false_positives,
        fpr,
    )) = results.last()
    {
        if max_elements > 0 {
            println!("\n🎯 Key Findings (at {} elements):", max_elements);
            println!("   Creation time: {:.6}s", creation_time);
            println!("   Insert rate: {:.0} ops/s", insert_rate);
            println!("   Query rate: {:.0} ops/s", query_rate);
            println!("   False positive rate: {:.4}", fpr);

            println!("\n💡 This is the native Rust performance without Python FFI overhead!");
        }
    }
}
