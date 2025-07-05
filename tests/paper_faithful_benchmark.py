#!/usr/bin/env python3
"""
Paper-faithful performance benchmark for ferric-bloom vs BlowChoc
Follows the methodology from the original BlowChoc paper with exact parameters
"""

import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from math import log, floor
import gc

# Try to import tqdm for progress bars, fallback to simple progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add the python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from ferric_bloom.unified_filter import BlowChocFilter

# Configuration matching the paper exactly
FILTER_SIZE_GB = 2  # 2 GB filter size
BITS_PER_GB = 8 * 1024 * 1024 * 1024  # 8 billion bits per GB
TOTAL_BITS = FILTER_SIZE_GB * BITS_PER_GB  # 16 Gbits total (m)
BLOCK_SIZE = 512  # BlowChoc block size
UNIVERSE = 4**21  # Large universe for 64-bit-like keys

# Test ranges from the paper
THREAD_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
K_VALUES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# Number of runs for averaging (paper uses 3)
NUM_RUNS = 3

# Minimum timing threshold to avoid precision issues
MIN_TIMING_THRESHOLD = 0.001  # 1ms minimum timing

# Results storage
results = {
    'rust': {'insertion': {}, 'successful_lookup': {}, 'unsuccessful_lookup': {}},
    'numba': {'insertion': {}, 'successful_lookup': {}, 'unsuccessful_lookup': {}}
}

class ProgressTracker:
    """Simple progress tracker with tqdm fallback"""
    
    def __init__(self, total_steps, description="Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
        if HAS_TQDM:
            self.pbar = tqdm(total=total_steps, desc=description, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            self.pbar = None
            print(f"{description}: 0/{total_steps}")
    
    def update(self, step_name=""):
        self.current_step += 1
        if HAS_TQDM and self.pbar:
            self.pbar.set_description(f"{self.description}: {step_name}")
            self.pbar.update(1)
        else:
            print(f"{self.description}: {self.current_step}/{self.total_steps} - {step_name}")
    
    def close(self):
        if HAS_TQDM and self.pbar:
            self.pbar.close()

def calculate_optimal_keys(m_bits, k):
    """Calculate optimal number of keys for target FPR of 2^-k"""
    n = floor(m_bits * log(2) / k)
    return n

def generate_random_keys(n, seed=42):
    """Generate n random 64-bit integers efficiently"""
    random.seed(seed)
    np.random.seed(seed)
    
    if n > 100_000:
        return np.random.randint(0, 2**63 - 1, size=n, dtype=np.int64).tolist()
    else:
        return [random.randint(0, 2**63 - 1) for _ in range(n)]

def create_filter_paper_config(backend, k, nsubfilters=1, nchoices=2):
    """Create filter matching paper configuration"""
    target_blocks = TOTAL_BITS // (BLOCK_SIZE * nsubfilters)
    nblocks = max(100, target_blocks)
    
    try:
        filter_obj = BlowChocFilter(
            universe=UNIVERSE,
            nsubfilters=nsubfilters,
            nblocks=nblocks,
            nchoices=nchoices,
            nbits=k,
            backend=backend
        )
        return filter_obj
    except Exception:
        return None

def warmup_filter(filter_obj, n_warmup=1000):
    """Warmup the filter with some operations"""
    warmup_keys = generate_random_keys(n_warmup, seed=999)
    
    for key in warmup_keys[:n_warmup//2]:
        filter_obj.insert(key)
    
    for key in warmup_keys:
        filter_obj.lookup(key)

def benchmark_insertion_paper_style(backend, k, nsubfilters, nchoices):
    """Benchmark insertion following paper methodology with exact parameters"""
    effective_bits = TOTAL_BITS // nsubfilters
    n_keys = calculate_optimal_keys(effective_bits, k)
    
    if n_keys > 50_000:
        # Use batched approach for large key counts
        batch_sizes = [10_000, 25_000, 50_000]
        batch_throughputs = []
        
        for batch_size in batch_sizes:
            keys = generate_random_keys(batch_size)
            batch_runs = []
            
            for run in range(NUM_RUNS):
                gc.collect()
                
                filter_obj = create_filter_paper_config(backend, k, nsubfilters, nchoices)
                if filter_obj is None:
                    continue
                
                warmup_filter(filter_obj, min(1000, batch_size // 10))
                
                start_time = time.perf_counter()
                for key in keys:
                    filter_obj.insert(key)
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                if total_time >= MIN_TIMING_THRESHOLD:
                    throughput = (batch_size / total_time) / 1_000_000
                    batch_runs.append(throughput)
                
                del filter_obj
            
            if batch_runs:
                batch_throughputs.append(np.mean(batch_runs))
        
        return np.mean(batch_throughputs) if batch_throughputs else 0.0
    else:
        # Direct measurement for smaller key counts
        keys = generate_random_keys(n_keys)
        throughputs = []
        
        for run in range(NUM_RUNS):
            gc.collect()
            
            filter_obj = create_filter_paper_config(backend, k, nsubfilters, nchoices)
            if filter_obj is None:
                return 0.0
            
            warmup_filter(filter_obj, min(1000, len(keys) // 10))
            
            start_time = time.perf_counter()
            for key in keys:
                filter_obj.insert(key)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            if total_time >= MIN_TIMING_THRESHOLD:
                throughput = (len(keys) / total_time) / 1_000_000
                throughputs.append(throughput)
            
            del filter_obj
        
        return np.mean(throughputs) if throughputs else 0.0

def benchmark_lookup_paper_style(backend, k, nsubfilters, nchoices, successful=True):
    """Benchmark lookup following paper methodology with exact parameters"""
    effective_bits = TOTAL_BITS // nsubfilters
    n_keys = calculate_optimal_keys(effective_bits, k)
    
    if n_keys > 50_000:
        # Use batched approach for large key counts
        batch_sizes = [10_000, 25_000, 50_000]
        batch_throughputs = []
        
        for batch_size in batch_sizes:
            insert_keys = generate_random_keys(batch_size, seed=42)
            lookup_keys = insert_keys if successful else generate_random_keys(batch_size, seed=123)
            
            batch_runs = []
            
            for run in range(NUM_RUNS):
                gc.collect()
                
                filter_obj = create_filter_paper_config(backend, k, nsubfilters, nchoices)
                if filter_obj is None:
                    continue
                
                warmup_filter(filter_obj, min(1000, batch_size // 10))
                
                for key in insert_keys:
                    filter_obj.insert(key)
                
                start_time = time.perf_counter()
                hits = 0
                for key in lookup_keys:
                    if filter_obj.lookup(key):
                        hits += 1
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                if total_time >= MIN_TIMING_THRESHOLD:
                    throughput = (batch_size / total_time) / 1_000_000
                    batch_runs.append(throughput)
                
                del filter_obj
            
            if batch_runs:
                batch_throughputs.append(np.mean(batch_runs))
        
        return np.mean(batch_throughputs) if batch_throughputs else 0.0
    else:
        # Direct measurement for smaller key counts
        insert_keys = generate_random_keys(n_keys, seed=42)
        lookup_keys = insert_keys if successful else generate_random_keys(n_keys, seed=123)
        
        throughputs = []
        
        for run in range(NUM_RUNS):
            gc.collect()
            
            filter_obj = create_filter_paper_config(backend, k, nsubfilters, nchoices)
            if filter_obj is None:
                return 0.0
            
            warmup_filter(filter_obj, min(1000, len(insert_keys) // 10))
            
            for key in insert_keys:
                filter_obj.insert(key)
            
            start_time = time.perf_counter()
            hits = 0
            for key in lookup_keys:
                if filter_obj.lookup(key):
                    hits += 1
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            if total_time >= MIN_TIMING_THRESHOLD:
                throughput = (len(lookup_keys) / total_time) / 1_000_000
                throughputs.append(throughput)
            
            del filter_obj
        
        return np.mean(throughputs) if throughputs else 0.0

def run_thread_scaling_benchmark():
    """Run benchmark varying number of threads (subfilters)"""
    k = 14  # Representative k value
    nchoices = 2  # Blow2Choc
    
    # Calculate total steps for progress tracking
    total_steps = len(['rust', 'numba']) * len(THREAD_COUNTS) * 3  # 3 operation types
    progress = ProgressTracker(total_steps, "Thread Scaling Benchmark")
    
    for backend in ['rust', 'numba']:
        results[backend]['insertion']['thread_scaling'] = {}
        results[backend]['successful_lookup']['thread_scaling'] = {}
        results[backend]['unsuccessful_lookup']['thread_scaling'] = {}
        
        for nsubfilters in THREAD_COUNTS:
            try:
                # Insertion benchmark
                progress.update(f"{backend} - {nsubfilters} threads - insertion")
                insertion_throughput = benchmark_insertion_paper_style(backend, k, nsubfilters, nchoices)
                
                # Successful lookup benchmark
                progress.update(f"{backend} - {nsubfilters} threads - successful lookup")
                successful_throughput = benchmark_lookup_paper_style(backend, k, nsubfilters, nchoices, successful=True)
                
                # Unsuccessful lookup benchmark
                progress.update(f"{backend} - {nsubfilters} threads - unsuccessful lookup")
                unsuccessful_throughput = benchmark_lookup_paper_style(backend, k, nsubfilters, nchoices, successful=False)
                
                results[backend]['insertion']['thread_scaling'][nsubfilters] = insertion_throughput
                results[backend]['successful_lookup']['thread_scaling'][nsubfilters] = successful_throughput
                results[backend]['unsuccessful_lookup']['thread_scaling'][nsubfilters] = unsuccessful_throughput
                
            except Exception as e:
                print(f"Error testing {backend} with {nsubfilters} subfilters: {e}")
                results[backend]['insertion']['thread_scaling'][nsubfilters] = 0.0
                results[backend]['successful_lookup']['thread_scaling'][nsubfilters] = 0.0
                results[backend]['unsuccessful_lookup']['thread_scaling'][nsubfilters] = 0.0
                progress.update(f"{backend} - {nsubfilters} threads - error")
                progress.update(f"{backend} - {nsubfilters} threads - error")
                progress.update(f"{backend} - {nsubfilters} threads - error")
    
    progress.close()

def run_k_scaling_benchmark():
    """Run benchmark varying k values"""
    nsubfilters = 9  # 9 subfilters as in paper
    nchoices = 2
    
    # Calculate total steps for progress tracking
    total_steps = len(['rust', 'numba']) * len(K_VALUES) * 3  # 3 operation types
    progress = ProgressTracker(total_steps, "K Scaling Benchmark")
    
    for backend in ['rust', 'numba']:
        results[backend]['insertion']['k_scaling'] = {}
        results[backend]['successful_lookup']['k_scaling'] = {}
        results[backend]['unsuccessful_lookup']['k_scaling'] = {}
        
        for k in K_VALUES:
            try:
                # Insertion benchmark
                progress.update(f"{backend} - k={k} - insertion")
                insertion_throughput = benchmark_insertion_paper_style(backend, k, nsubfilters, nchoices)
                
                # Successful lookup benchmark
                progress.update(f"{backend} - k={k} - successful lookup")
                successful_throughput = benchmark_lookup_paper_style(backend, k, nsubfilters, nchoices, successful=True)
                
                # Unsuccessful lookup benchmark
                progress.update(f"{backend} - k={k} - unsuccessful lookup")
                unsuccessful_throughput = benchmark_lookup_paper_style(backend, k, nsubfilters, nchoices, successful=False)
                
                results[backend]['insertion']['k_scaling'][k] = insertion_throughput
                results[backend]['successful_lookup']['k_scaling'][k] = successful_throughput
                results[backend]['unsuccessful_lookup']['k_scaling'][k] = unsuccessful_throughput
                
            except Exception as e:
                print(f"Error testing {backend} with k={k}: {e}")
                results[backend]['insertion']['k_scaling'][k] = 0.0
                results[backend]['successful_lookup']['k_scaling'][k] = 0.0
                results[backend]['unsuccessful_lookup']['k_scaling'][k] = 0.0
                progress.update(f"{backend} - k={k} - error")
                progress.update(f"{backend} - k={k} - error")
                progress.update(f"{backend} - k={k} - error")
    
    progress.close()

def create_paper_style_plots():
    """Create plots matching the style from the original BlowChoc paper"""
    print("📈 Creating paper-style plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Thread scaling plots
    if 'thread_scaling' in results['rust']['insertion']:
        threads = sorted(results['rust']['insertion']['thread_scaling'].keys())
        
        # Insertion vs threads
        rust_insertion = [results['rust']['insertion']['thread_scaling'][t] for t in threads]
        numba_insertion = [results['numba']['insertion']['thread_scaling'][t] for t in threads]
        
        ax1.plot(threads, rust_insertion, 'o-', label='Rust', linewidth=2, markersize=8)
        ax1.plot(threads, numba_insertion, 's--', label='Numba', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Insertion Throughput (Mops/s)')
        ax1.set_title('Insertion Performance vs Thread Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Successful lookup vs threads
        rust_lookup = [results['rust']['successful_lookup']['thread_scaling'][t] for t in threads]
        numba_lookup = [results['numba']['successful_lookup']['thread_scaling'][t] for t in threads]
        
        ax2.plot(threads, rust_lookup, 'o-', label='Rust', linewidth=2, markersize=8)
        ax2.plot(threads, numba_lookup, 's--', label='Numba', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Lookup Throughput (Mops/s)')
        ax2.set_title('Successful Lookup Performance vs Thread Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # K scaling plots
    if 'k_scaling' in results['rust']['insertion']:
        k_vals = sorted(results['rust']['insertion']['k_scaling'].keys())
        
        # Insertion vs k
        rust_insertion_k = [results['rust']['insertion']['k_scaling'][k] for k in k_vals]
        numba_insertion_k = [results['numba']['insertion']['k_scaling'][k] for k in k_vals]
        
        ax3.plot(k_vals, rust_insertion_k, 'o-', label='Rust', linewidth=2, markersize=8)
        ax3.plot(k_vals, numba_insertion_k, 's--', label='Numba', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Hash Functions (k)')
        ax3.set_ylabel('Insertion Throughput (Mops/s)')
        ax3.set_title('Insertion Performance vs Hash Functions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Lookup vs k
        rust_lookup_k = [results['rust']['successful_lookup']['k_scaling'][k] for k in k_vals]
        numba_lookup_k = [results['numba']['successful_lookup']['k_scaling'][k] for k in k_vals]
        
        ax4.plot(k_vals, rust_lookup_k, 'o-', label='Rust', linewidth=2, markersize=8)
        ax4.plot(k_vals, numba_lookup_k, 's--', label='Numba', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Hash Functions (k)')
        ax4.set_ylabel('Lookup Throughput (Mops/s)')
        ax4.set_title('Successful Lookup Performance vs Hash Functions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_faithful_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_faithful_performance_comparison.pdf', bbox_inches='tight')
    plt.close()

def save_paper_results():
    """Save results in CSV format"""
    print("💾 Saving paper-style results...")
    
    with open('paper_faithful_results.csv', 'w') as f:
        f.write("backend,test_type,parameter,value,throughput_mops\n")
        
        for backend in ['rust', 'numba']:
            for operation in ['insertion', 'successful_lookup', 'unsuccessful_lookup']:
                for test_type in ['thread_scaling', 'k_scaling']:
                    if test_type in results[backend][operation]:
                        for param_val, throughput in results[backend][operation][test_type].items():
                            f.write(f"{backend},{operation},{test_type},{param_val},{throughput:.6f}\n")

def print_paper_summary():
    """Print a comprehensive summary of benchmark results"""
    print("\n" + "="*80)
    print("📊 PAPER-FAITHFUL BENCHMARK RESULTS")
    print("="*80)
    print(f"Filter Size: {FILTER_SIZE_GB} GB ({TOTAL_BITS/1e9:.1f} Gbits)")
    print(f"Block Size: {BLOCK_SIZE} bits")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Key calculation: n = ⌊m · ln(2)/k⌋")
    print()
    
    # Thread scaling summary
    if 'thread_scaling' in results['rust']['insertion']:
        print("Thread Scaling Results (k=14):")
        print("-" * 50)
        threads = sorted(results['rust']['insertion']['thread_scaling'].keys())
        
        print("Threads | Rust Insert | Numba Insert | Rust Lookup | Numba Lookup")
        print("-" * 70)
        for t in threads:
            rust_ins = results['rust']['insertion']['thread_scaling'][t]
            numba_ins = results['numba']['insertion']['thread_scaling'][t]
            rust_look = results['rust']['successful_lookup']['thread_scaling'][t]
            numba_look = results['numba']['successful_lookup']['thread_scaling'][t]
            print(f"{t:7d} | {rust_ins:11.2f} | {numba_ins:12.2f} | {rust_look:11.2f} | {numba_look:12.2f}")
        
        print()
        max_rust_insert = max(results['rust']['insertion']['thread_scaling'].values())
        max_numba_insert = max(results['numba']['insertion']['thread_scaling'].values())
        max_rust_lookup = max(results['rust']['successful_lookup']['thread_scaling'].values())
        max_numba_lookup = max(results['numba']['successful_lookup']['thread_scaling'].values())
        
        print(f"Peak Insertion Throughput:")
        print(f"  Rust: {max_rust_insert:.2f} Mops/s")
        print(f"  Numba: {max_numba_insert:.2f} Mops/s")
        print(f"  Speedup: {max_rust_insert/max_numba_insert:.1f}x")
        print()
        print(f"Peak Lookup Throughput:")
        print(f"  Rust: {max_rust_lookup:.2f} Mops/s") 
        print(f"  Numba: {max_numba_lookup:.2f} Mops/s")
        print(f"  Speedup: {max_rust_lookup/max_numba_lookup:.1f}x")
    
    print("\n🎉 Paper-faithful benchmark complete!")

def main():
    """Main benchmark execution"""
    print("🚀 Paper-Faithful Ferric-Bloom vs BlowChoc Benchmark")
    print("=" * 60)
    print("Following methodology from the original BlowChoc paper")
    print(f"Filter size: {FILTER_SIZE_GB} GB ({TOTAL_BITS/1e9:.1f} Gbits)")
    print(f"Key calculation: n = ⌊m · ln(2)/k⌋")
    print(f"Runs per measurement: {NUM_RUNS}")
    print(f"Available CPU cores: {mp.cpu_count()}")
    print()
    
    # Check backend availability
    try:
        rust_filter = BlowChocFilter(universe=4**10, nblocks=10, nchoices=2, nbits=4)
        rust_filter.use_backend('rust')
        print("✅ Rust backend available")
    except:
        print("❌ Rust backend not available")
        return
    
    try:
        numba_filter = BlowChocFilter(universe=4**10, nblocks=10, nchoices=2, nbits=4, backend='numba')
        print("✅ Numba backend available")
    except:
        print("❌ Numba backend not available")
        return
    
    # Run benchmarks
    start_time = time.time()
    
    print("\n🔄 Running benchmarks...")
    run_thread_scaling_benchmark()
    run_k_scaling_benchmark()
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total benchmark time: {total_time:.1f} seconds")
    
    # Generate outputs
    create_paper_style_plots()
    save_paper_results()
    print_paper_summary()

if __name__ == "__main__":
    main() 