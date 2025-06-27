#!/usr/bin/env python3
"""
Compare ferric-bloom (native Rust) vs blowchoc (Python+Numba).
This eliminates Python FFI overhead from ferric-bloom measurements.
"""

import sys
import os
import time
import gc
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Importing blowchoc...")
try:
    from blowchoc.filter_bb_choices import build_filter as build_blocked_filter
    BLOWCHOC_AVAILABLE = True
    print("✅ blowchoc imported successfully")
except ImportError as e:
    BLOWCHOC_AVAILABLE = False
    print(f"❌ blowchoc import failed: {e}")

def run_ferric_benchmark() -> pd.DataFrame:
    """Run the ferric-bloom native Rust benchmark and parse results"""
    print("\n🦀 Running ferric-bloom native benchmark...")
    
    # Change to project root to run cargo
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("   Compiling and running ferric-bloom benchmark...")
        # Run the Rust benchmark
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "rust_micro_benchmark"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ ferric-bloom benchmark failed: {result.stderr}")
            print(f"   stdout: {result.stdout}")
            return pd.DataFrame()
        
        print("   Parsing results...")
        # Parse the CSV output
        lines = result.stdout.split('\n')
        csv_start = False
        csv_data = []
        
        for line in lines:
            if line.startswith('elements,creation_time'):
                csv_start = True
                continue
            elif csv_start and line.strip() and ',' in line:
                csv_data.append(line.strip())
            elif csv_start and not line.strip():
                break
        
        if not csv_data:
            print("❌ No CSV data found in ferric-bloom output")
            print("Full output:")
            print(result.stdout)
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for line in csv_data:
            parts = line.split(',')
            if len(parts) >= 9:
                try:
                    data.append({
                        'elements': int(parts[0]),
                        'creation_time': float(parts[1]),
                        'insert_time': float(parts[2]),
                        'query_time': float(parts[3]),
                        'insert_rate': float(parts[4]),
                        'query_rate': float(parts[5]),
                        'hits': int(parts[6]),
                        'false_positives': int(parts[7]),
                        'false_positive_rate': float(parts[8])
                    })
                except ValueError as e:
                    print(f"⚠️  Error parsing line: {line} - {e}")
                    continue
        
        df = pd.DataFrame(data)
        print(f"✅ ferric-bloom benchmark completed - {len(df)} data points")
        return df
        
    except subprocess.TimeoutExpired:
        print("❌ ferric-bloom benchmark timed out")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error running ferric-bloom benchmark: {e}")
        return pd.DataFrame()

def run_blowchoc_benchmark() -> pd.DataFrame:
    """Run blowchoc benchmark"""
    print("\n🐍 Running blowchoc benchmark...")
    
    if not BLOWCHOC_AVAILABLE:
        print("❌ blowchoc not available")
        return pd.DataFrame()
    
    element_counts = list(range(0, 201, 25))
    results = []
    
    for i, n_elements in enumerate(element_counts):
        print(f"   Testing {n_elements} elements... ({i+1}/{len(element_counts)})")
        
        if n_elements == 0:
            results.append({
                'elements': 0,
                'creation_time': 0.0,
                'insert_time': 0.0,
                'query_time': 0.0,
                'insert_rate': 0.0,
                'query_rate': 0.0,
                'hits': 0,
                'false_positives': 0,
                'false_positive_rate': 0.0
            })
            continue
        
        try:
            # Configuration (same as ferric-bloom benchmark)
            nblocks = 100
            bits_per_key = 8
            nchoices = 2
            
            # Blowchoc parameters (conservative to avoid hanging)
            universe = 2**20
            nsubfilters = 3
            sigma = 1.0
            theta = 1.0
            beta = 1.618
            
            # Create filter
            start_time = time.perf_counter()
            blowchoc_filter = build_blocked_filter(
                universe=universe,
                nsubfilters=nsubfilters,
                nblocks=nblocks,
                nchoices=nchoices,
                nbits=bits_per_key,
                sigma=sigma,
                theta=theta,
                beta=beta
            )
            creation_time = time.perf_counter() - start_time
            
            # Generate test data
            test_keys = list(range(n_elements))
            query_keys = list(range(n_elements, n_elements + min(100, n_elements)))
            
            bit_positions = np.empty(bits_per_key, dtype=np.uint64)
            
            # Insert benchmark
            start_time = time.perf_counter()
            for key in test_keys:
                blowchoc_filter.insert(blowchoc_filter.array, key, bit_positions)
            insert_time = time.perf_counter() - start_time
            
            # Query benchmark (successful lookups)
            sample_keys = test_keys[:min(10, len(test_keys))]
            hits = 0
            for key in sample_keys:
                if blowchoc_filter.lookup(blowchoc_filter.array, key, bit_positions):
                    hits += 1
            
            # Query benchmark (false positive test)
            start_time = time.perf_counter()
            false_positives = 0
            for key in query_keys:
                if blowchoc_filter.lookup(blowchoc_filter.array, key, bit_positions):
                    false_positives += 1
            fp_query_time = time.perf_counter() - start_time
            
            # Calculate rates
            insert_rate = n_elements / insert_time if insert_time > 0 else float('inf')
            query_rate = len(query_keys) / fp_query_time if fp_query_time > 0 else float('inf')
            false_positive_rate = false_positives / len(query_keys) if query_keys else 0.0
            
            results.append({
                'elements': n_elements,
                'creation_time': creation_time,
                'insert_time': insert_time,
                'query_time': fp_query_time,
                'insert_rate': insert_rate,
                'query_rate': query_rate,
                'hits': hits,
                'false_positives': false_positives,
                'false_positive_rate': false_positive_rate
            })
            
            print(f"      ✅ Insert: {insert_rate:,.0f} ops/s, Query: {query_rate:,.0f} ops/s")
            
        except Exception as e:
            print(f"      ❌ Error at {n_elements} elements: {e}")
            # Add dummy data to keep going
            results.append({
                'elements': n_elements,
                'creation_time': float('inf'),
                'insert_time': float('inf'),
                'query_time': float('inf'),
                'insert_rate': 0.0,
                'query_rate': 0.0,
                'hits': 0,
                'false_positives': 0,
                'false_positive_rate': 0.0
            })
        
        gc.collect()
    
    df = pd.DataFrame(results)
    print(f"✅ blowchoc benchmark completed - {len(df)} data points")
    return df

def create_comparison_plots(ferric_df: pd.DataFrame, blowchoc_df: pd.DataFrame):
    """Create comparison plots between ferric-bloom and blowchoc implementations"""
    
    if ferric_df.empty or blowchoc_df.empty:
        print("❌ Cannot create plots - missing data")
        return
    
    print("📊 Creating comparison plots...")
    
    # Configure matplotlib for publication-quality plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Colors
    ferric_color = '#D55E00'  # Orange
    blowchoc_color = '#0072B2'  # Blue
    
    # Filter out zero elements and invalid data for log plots
    ferric_nonzero = ferric_df[(ferric_df['elements'] > 0) & (ferric_df['insert_rate'] > 0) & (ferric_df['query_rate'] > 0)]
    blowchoc_nonzero = blowchoc_df[(blowchoc_df['elements'] > 0) & (blowchoc_df['insert_rate'] > 0) & (blowchoc_df['query_rate'] > 0)]
    
    # Plot 1: Insert Throughput
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not ferric_nonzero.empty:
        ax.plot(ferric_nonzero['elements'], ferric_nonzero['insert_rate'], 'o-', 
               color=ferric_color, linewidth=1.5, markersize=4, label='ferric-bloom')
    if not blowchoc_nonzero.empty:
        ax.plot(blowchoc_nonzero['elements'], blowchoc_nonzero['insert_rate'], 's-', 
               color=blowchoc_color, linewidth=1.5, markersize=4, label='blowchoc')
    
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Insert Rate (ops/s)')
    ax.set_title('Insert Throughput Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ferric_vs_blowchoc_insert_throughput.pdf', format='pdf')
    plt.savefig('ferric_vs_blowchoc_insert_throughput.png', format='png', dpi=300)
    plt.close()
    
    # Plot 2: Query Throughput
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not ferric_nonzero.empty:
        ax.plot(ferric_nonzero['elements'], ferric_nonzero['query_rate'], 'o-', 
               color=ferric_color, linewidth=1.5, markersize=4, label='ferric-bloom')
    if not blowchoc_nonzero.empty:
        ax.plot(blowchoc_nonzero['elements'], blowchoc_nonzero['query_rate'], 's-', 
               color=blowchoc_color, linewidth=1.5, markersize=4, label='blowchoc')
    
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Query Rate (ops/s)')
    ax.set_title('Query Throughput Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ferric_vs_blowchoc_query_throughput.pdf', format='pdf')
    plt.savefig('ferric_vs_blowchoc_query_throughput.png', format='png', dpi=300)
    plt.close()
    
    # Plot 3: Insert Time
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not ferric_nonzero.empty:
        ax.plot(ferric_nonzero['elements'], ferric_nonzero['insert_time'] * 1000, 'o-', 
               color=ferric_color, linewidth=1.5, markersize=4, label='ferric-bloom')
    if not blowchoc_nonzero.empty:
        ax.plot(blowchoc_nonzero['elements'], blowchoc_nonzero['insert_time'] * 1000, 's-', 
               color=blowchoc_color, linewidth=1.5, markersize=4, label='blowchoc')
    
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Insert Time (ms)')
    ax.set_title('Insert Time Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ferric_vs_blowchoc_insert_time.pdf', format='pdf')
    plt.savefig('ferric_vs_blowchoc_insert_time.png', format='png', dpi=300)
    plt.close()
    
    # Plot 4: Query Time
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not ferric_nonzero.empty:
        ax.plot(ferric_nonzero['elements'], ferric_nonzero['query_time'] * 1000, 'o-', 
               color=ferric_color, linewidth=1.5, markersize=4, label='ferric-bloom')
    if not blowchoc_nonzero.empty:
        ax.plot(blowchoc_nonzero['elements'], blowchoc_nonzero['query_time'] * 1000, 's-', 
               color=blowchoc_color, linewidth=1.5, markersize=4, label='blowchoc')
    
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Query Time (ms)')
    ax.set_title('Query Time Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ferric_vs_blowchoc_query_time.pdf', format='pdf')
    plt.savefig('ferric_vs_blowchoc_query_time.png', format='png', dpi=300)
    plt.close()
    
    # Plot 5: Creation Time
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not ferric_nonzero.empty:
        ax.plot(ferric_nonzero['elements'], ferric_nonzero['creation_time'] * 1000, 'o-', 
               color=ferric_color, linewidth=1.5, markersize=4, label='ferric-bloom')
    if not blowchoc_nonzero.empty:
        ax.plot(blowchoc_nonzero['elements'], blowchoc_nonzero['creation_time'] * 1000, 's-', 
               color=blowchoc_color, linewidth=1.5, markersize=4, label='blowchoc')
    
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Creation Time (ms)')
    ax.set_title('Filter Creation Time Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ferric_vs_blowchoc_creation_time.pdf', format='pdf')
    plt.savefig('ferric_vs_blowchoc_creation_time.png', format='png', dpi=300)
    plt.close()
    
    print("📄 Generated comparison plots:")
    print("   • ferric_vs_blowchoc_insert_throughput.pdf/.png")
    print("   • ferric_vs_blowchoc_query_throughput.pdf/.png")
    print("   • ferric_vs_blowchoc_insert_time.pdf/.png")
    print("   • ferric_vs_blowchoc_query_time.pdf/.png")
    print("   • ferric_vs_blowchoc_creation_time.pdf/.png")
    print("💾 Generated CSV data files:")
    print("   • ferric_bloom_results.csv")
    print("   • blowchoc_results.csv") 
    print("   • ferric_vs_blowchoc_comparison.csv")

def save_results_to_csv(ferric_df: pd.DataFrame, blowchoc_df: pd.DataFrame):
    """Save benchmark results to CSV files"""
    
    if ferric_df.empty and blowchoc_df.empty:
        print("❌ No data to save")
        return
    
    print("💾 Saving results to CSV files...")
    
    # Save ferric-bloom results
    if not ferric_df.empty:
        ferric_df.to_csv('ferric_bloom_results.csv', index=False)
        print("   • ferric_bloom_results.csv")
    
    # Save blowchoc results
    if not blowchoc_df.empty:
        blowchoc_df.to_csv('blowchoc_results.csv', index=False)
        print("   • blowchoc_results.csv")
    
    # Create combined comparison CSV
    if not ferric_df.empty and not blowchoc_df.empty:
        # Merge on elements column
        combined = pd.merge(ferric_df, blowchoc_df, on='elements', suffixes=('_ferric', '_blowchoc'))
        
        # Add speedup columns
        combined['insert_speedup'] = combined['insert_rate_ferric'] / combined['insert_rate_blowchoc']
        combined['query_speedup'] = combined['query_rate_ferric'] / combined['query_rate_blowchoc']
        combined['insert_time_speedup'] = combined['insert_time_blowchoc'] / combined['insert_time_ferric']
        combined['query_time_speedup'] = combined['query_time_blowchoc'] / combined['query_time_ferric']
        
        # Replace inf values with a large number for CSV readability
        combined = combined.replace([float('inf'), -float('inf')], 999999)
        
        combined.to_csv('ferric_vs_blowchoc_comparison.csv', index=False)
        print("   • ferric_vs_blowchoc_comparison.csv")
    
    print("📊 CSV files saved successfully!")

def print_comparison_table(ferric_df: pd.DataFrame, blowchoc_df: pd.DataFrame):
    """Print a comparison table of key metrics"""
    
    if ferric_df.empty or blowchoc_df.empty:
        print("❌ Cannot create comparison - missing data")
        return
    
    print("\n📊 Performance Comparison (ferric-bloom vs blowchoc)")
    print("=" * 65)
    print(f"{'Elements':<8} {'Metric':<15} {'ferric-bloom':<15} {'blowchoc':<15} {'Speedup':<10}")
    print("-" * 65)
    
    # Compare at key element counts
    test_counts = [25, 50, 100, 200]
    
    for n in test_counts:
        ferric_row = ferric_df[ferric_df['elements'] == n]
        blowchoc_row = blowchoc_df[blowchoc_df['elements'] == n]
        
        if ferric_row.empty or blowchoc_row.empty:
            continue
            
        ferric_data = ferric_row.iloc[0]
        blowchoc_data = blowchoc_row.iloc[0]
        
        # Skip invalid data
        if ferric_data['insert_rate'] == 0 or blowchoc_data['insert_rate'] == 0:
            continue
            
        # Insert rate
        insert_speedup = ferric_data['insert_rate'] / blowchoc_data['insert_rate'] if blowchoc_data['insert_rate'] > 0 else float('inf')
        print(f"{n:<8} {'Insert (ops/s)':<15} {ferric_data['insert_rate']:<15.0f} {blowchoc_data['insert_rate']:<15.0f} {insert_speedup:<10.1f}x")
        
        # Query rate
        query_speedup = ferric_data['query_rate'] / blowchoc_data['query_rate'] if blowchoc_data['query_rate'] > 0 else float('inf')
        print(f"{n:<8} {'Query (ops/s)':<15} {ferric_data['query_rate']:<15.0f} {blowchoc_data['query_rate']:<15.0f} {query_speedup:<10.1f}x")
        
        # Insert time
        insert_time_speedup = blowchoc_data['insert_time'] / ferric_data['insert_time'] if ferric_data['insert_time'] > 0 else float('inf')
        print(f"{n:<8} {'Insert (ms)':<15} {ferric_data['insert_time']*1000:<15.3f} {blowchoc_data['insert_time']*1000:<15.0f} {insert_time_speedup:<10.1f}x")
        
        # Query time
        query_time_speedup = blowchoc_data['query_time'] / ferric_data['query_time'] if ferric_data['query_time'] > 0 else float('inf')
        print(f"{n:<8} {'Query (ms)':<15} {ferric_data['query_time']*1000:<15.3f} {blowchoc_data['query_time']*1000:<15.3f} {query_time_speedup:<10.1f}x")
        
        print()

def main():
    """Run the comparison benchmark"""
    print("🔬 ferric-bloom vs blowchoc Benchmark Comparison")
    print("Comparing ferric-bloom (native Rust) vs blowchoc (Python+Numba)")
    print("=" * 65)
    
    start_time = time.time()
    
    # Run benchmarks
    ferric_results = run_ferric_benchmark()
    blowchoc_results = run_blowchoc_benchmark()
    
    if ferric_results.empty:
        print("❌ Failed to get ferric-bloom results")
        return
    
    if blowchoc_results.empty:
        print("❌ Failed to get blowchoc results")
        return
    
    # Save results to CSV files
    save_results_to_csv(ferric_results, blowchoc_results)
    
    # Create comparison plots
    create_comparison_plots(ferric_results, blowchoc_results)
    
    # Print comparison table
    print_comparison_table(ferric_results, blowchoc_results)
    
    total_time = time.time() - start_time
    print(f"\n✅ Comparison benchmark completed in {total_time:.1f} seconds!")
    
    # Print key insights
    valid_ferric = ferric_results[(ferric_results['elements'] == 200) & (ferric_results['insert_rate'] > 0)]
    valid_blowchoc = blowchoc_results[(blowchoc_results['elements'] == 200) & (blowchoc_results['insert_rate'] > 0)]
    
    if not valid_ferric.empty and not valid_blowchoc.empty:
        ferric_max = valid_ferric.iloc[0]
        blowchoc_max = valid_blowchoc.iloc[0]
        
        print(f"\n💡 Key Insights (at 200 elements):")
        print(f"   🦀 ferric-bloom (native Rust) vs blowchoc (Python+Numba)")
        print(f"   ⚡ Insert speedup: {ferric_max['insert_rate'] / blowchoc_max['insert_rate']:.1f}x faster")
        print(f"   🔍 Query speedup: {ferric_max['query_rate'] / blowchoc_max['query_rate']:.1f}x faster")
        print(f"   ⏱️  Insert time: {ferric_max['insert_time']*1000:.3f}ms vs {blowchoc_max['insert_time']*1000:.0f}ms")
        print(f"   📊 This comparison eliminates Python binding overhead!")

if __name__ == "__main__":
    main() 