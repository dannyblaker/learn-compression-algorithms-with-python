"""
Run All Compression Methods and Generate CSV Report

This script runs all implemented compression algorithms on all test files
and generates a comprehensive CSV report comparing their performance.

The report includes:
- Original file sizes
- Compressed sizes for each method
- Compression ratios
- Compression times
- Best performing method for each file
"""

from compression_methods.burrows_wheeler import BWTCompressor
from compression_methods.arithmetic_compression import ArithmeticCompressor
from compression_methods.lzw_compression import LZWCompressor
from compression_methods.lz77_compression import LZ77Compressor
from compression_methods.huffman_compression import HuffmanCompressor
from compression_methods.rle_compression import RLECompressor
from utils.file_utils import (
    get_all_test_files, get_file_size, format_size,
    CompressionStats, calculate_compression_ratio
)
import os
import sys
import csv
import time
from typing import List, Dict, Any

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import all compression methods


class CompressionBenchmark:
    """Benchmark suite for all compression methods."""

    def __init__(self):
        self.compressors = {
            'RLE': RLECompressor(),
            'Huffman': HuffmanCompressor(),
            'LZ77': LZ77Compressor(),
            'LZW': LZWCompressor(),
            'Arithmetic': ArithmeticCompressor(),
            'BWT': BWTCompressor()
        }

        self.output_dirs = {
            'RLE': 'compressed_output/rle',
            'Huffman': 'compressed_output/huffman',
            'LZ77': 'compressed_output/lz77',
            'LZW': 'compressed_output/lzw',
            'Arithmetic': 'compressed_output/arithmetic',
            'BWT': 'compressed_output/bwt'
        }

        self.file_extensions = {
            'RLE': '.rle',
            'Huffman': '.huff',
            'LZ77': '.lz77',
            'LZW': '.lzw',
            'Arithmetic': '.arith',
            'BWT': '.bwt'
        }

    def setup_output_directories(self):
        """Create output directories for all compression methods."""
        for output_dir in self.output_dirs.values():
            os.makedirs(output_dir, exist_ok=True)

    def compress_file_with_method(self, input_file: str, method_name: str) -> CompressionStats:
        """
        Compress a single file with a specific method.

        Args:
            input_file: Path to input file
            method_name: Name of compression method

        Returns:
            CompressionStats object with results
        """
        compressor = self.compressors[method_name]
        output_dir = self.output_dirs[method_name]
        file_ext = self.file_extensions[method_name]

        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{filename}{file_ext}")

        try:
            original_size = get_file_size(input_file)

            # Compress file and measure time
            start_time = time.time()
            compressed_size = compressor.compress_file(
                input_file, output_file)[0]
            end_time = time.time()
            compression_time = end_time - start_time

            return CompressionStats(method_name, original_size, compressed_size, compression_time)

        except Exception as e:
            print(f"Error compressing {filename} with {method_name}: {e}")
            original_size = get_file_size(input_file)
            return CompressionStats(method_name, original_size, original_size, 0.0)

    def run_comprehensive_benchmark(self, test_dir: str = "files_to_compress") -> List[Dict[str, Any]]:
        """
        Run all compression methods on all test files.

        Args:
            test_dir: Directory containing test files

        Returns:
            List of dictionaries with comprehensive results
        """
        if not os.path.exists(test_dir):
            print(f"Error: Test directory '{test_dir}' not found!")
            return []

        self.setup_output_directories()
        test_files = get_all_test_files(test_dir)

        if not test_files:
            print(f"No test files found in {test_dir}")
            return []

        print(f"Starting comprehensive compression benchmark")
        print(f"Test files: {len(test_files)}")
        print(f"Compression methods: {len(self.compressors)}")
        print(f"Total operations: {len(test_files) * len(self.compressors)}")
        print("=" * 80)

        results = []

        for file_idx, input_file in enumerate(test_files):
            filename = os.path.basename(input_file)
            original_size = get_file_size(input_file)

            print(f"\nFile {file_idx + 1}/{len(test_files)}: {filename}")
            print(f"Original size: {format_size(original_size)}")
            print("-" * 60)

            file_results = {
                'filename': filename,
                'original_size': original_size,
                'original_size_formatted': format_size(original_size)
            }

            method_stats = {}

            # Test each compression method
            for method_name in self.compressors.keys():
                print(f"  Testing {method_name}... ", end="", flush=True)

                stats = self.compress_file_with_method(input_file, method_name)
                method_stats[method_name] = stats

                print(f"{format_size(stats.compressed_size)} "
                      f"({stats.compression_ratio:.1f}%) "
                      f"in {stats.compression_time:.2f}s")

                # Add method-specific data to results
                file_results[f'{method_name}_size'] = stats.compressed_size
                file_results[f'{method_name}_size_formatted'] = format_size(
                    stats.compressed_size)
                file_results[f'{method_name}_ratio'] = stats.compression_ratio
                file_results[f'{method_name}_time'] = stats.compression_time

            # Find best method for this file
            best_method = max(method_stats.keys(),
                              key=lambda m: method_stats[m].compression_ratio)
            best_stats = method_stats[best_method]

            file_results['best_method'] = best_method
            file_results['best_ratio'] = best_stats.compression_ratio
            file_results['best_size'] = best_stats.compressed_size

            print(
                f"  Best: {best_method} ({best_stats.compression_ratio:.1f}%)")

            results.append(file_results)

        return results

    def generate_csv_report(self, results: List[Dict[str, Any]], output_file: str = "compression_report.csv"):
        """
        Generate comprehensive CSV report.

        Args:
            results: Benchmark results
            output_file: Output CSV filename
        """
        if not results:
            print("No results to write to CSV")
            return

        print(f"\nGenerating CSV report: {output_file}")

        # Define CSV columns
        columns = [
            'filename',
            'original_size',
            'original_size_formatted'
        ]

        # Add columns for each method
        for method in self.compressors.keys():
            columns.extend([
                f'{method}_size',
                f'{method}_size_formatted',
                f'{method}_ratio',
                f'{method}_time'
            ])

        columns.extend([
            'best_method',
            'best_ratio',
            'best_size'
        ])

        # Write CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"CSV report saved: {output_file}")

    def print_summary_statistics(self, results: List[Dict[str, Any]]):
        """
        Print summary statistics for all methods.

        Args:
            results: Benchmark results
        """
        if not results:
            return

        print("\n" + "="*80)
        print("COMPRESSION BENCHMARK SUMMARY")
        print("="*80)

        total_original = sum(r['original_size'] for r in results)

        print(f"Files processed: {len(results)}")
        print(f"Total original size: {format_size(total_original)}")
        print()

        # Summary for each method
        print("Method Performance Summary:")
        print("-" * 50)

        method_summaries = {}

        for method in self.compressors.keys():
            total_compressed = sum(r[f'{method}_size'] for r in results)
            total_time = sum(r[f'{method}_time'] for r in results)
            overall_ratio = calculate_compression_ratio(
                total_original, total_compressed)

            # Count wins (best method for file)
            wins = sum(1 for r in results if r['best_method'] == method)

            method_summaries[method] = {
                'total_compressed': total_compressed,
                'overall_ratio': overall_ratio,
                'total_time': total_time,
                'wins': wins
            }

            print(f"{method:12}: {format_size(total_compressed):>10} "
                  f"({overall_ratio:>6.1f}%) "
                  f"{total_time:>8.2f}s "
                  f"{wins:>3} wins")

        # Find overall best method
        best_overall = max(method_summaries.keys(),
                           key=lambda m: method_summaries[m]['overall_ratio'])

        print()
        print(f"Best overall method: {best_overall} "
              f"({method_summaries[best_overall]['overall_ratio']:.1f}% compression)")

        # File type analysis
        print("\nFile Type Analysis:")
        print("-" * 50)

        file_types = {}
        for result in results:
            filename = result['filename']
            ext = os.path.splitext(filename)[1].lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(result)

        for file_type, type_results in file_types.items():
            if not file_type:
                file_type = "(no extension)"

            type_original = sum(r['original_size'] for r in type_results)

            # Find best method for this file type
            type_best_ratios = {}
            for method in self.compressors.keys():
                type_compressed = sum(r[f'{method}_size']
                                      for r in type_results)
                type_ratio = calculate_compression_ratio(
                    type_original, type_compressed)
                type_best_ratios[method] = type_ratio

            best_for_type = max(type_best_ratios.keys(),
                                key=lambda m: type_best_ratios[m])
            best_ratio = type_best_ratios[best_for_type]

            print(f"{file_type:12}: {len(type_results):>2} files, "
                  f"best method: {best_for_type} ({best_ratio:.1f}%)")


def main():
    """Main function to run all compression benchmarks."""
    print("Compression Algorithms Benchmark Suite")
    print("=" * 50)
    print()
    print("This will run all compression methods on all test files")
    print("and generate a comprehensive CSV report.")
    print()

    benchmark = CompressionBenchmark()

    # Run comprehensive benchmark
    start_time = time.time()
    results = benchmark.run_comprehensive_benchmark()
    end_time = time.time()

    if results:
        # Generate CSV report
        benchmark.generate_csv_report(results)

        # Print summary statistics
        benchmark.print_summary_statistics(results)

        print(f"\nTotal benchmark time: {end_time - start_time:.2f} seconds")
        print("\nBenchmark complete! Check 'compression_report.csv' for detailed results.")
        print("\nCompressed files are saved in 'compressed_output/' directory,")
        print("organized by compression method.")
    else:
        print("No results generated. Please check that test files exist.")


if __name__ == "__main__":
    main()
