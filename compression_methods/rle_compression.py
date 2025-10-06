"""
Run-Length Encoding (RLE) Compression

RLE is one of the simplest compression algorithms. It works by replacing
sequences of identical bytes with a count and the byte value.

Example: "AAABBBCCCC" becomes "3A3B4C"

This implementation shows the basic concepts of compression:
- Pattern recognition (repeated bytes)
- Encoding scheme (count + value)
- Trade-offs (may expand data with no repetition)
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RLECompressor:
    """Run-Length Encoding compressor implementation."""

    def __init__(self):
        self.name = "RLE"

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using Run-Length Encoding.

        Format: For each run, store:
        - Length byte (1-255, 0 means 256)
        - Value byte

        Args:
            data: Input bytes to compress

        Returns:
            Compressed bytes
        """
        if not data:
            return b''

        compressed = bytearray()
        i = 0

        print(f"Compressing {len(data)} bytes using RLE...")

        while i < len(data):
            current_byte = data[i]
            count = 1

            # Count consecutive identical bytes (max 256)
            while (i + count < len(data) and
                   data[i + count] == current_byte and
                   count < 256):
                count += 1

            # Store count and byte value
            # Use 0 to represent count of 256
            count_byte = 0 if count == 256 else count
            compressed.extend([count_byte, current_byte])

            i += count

            # Progress indicator for large files
            if i % 10000 == 0:
                print(
                    f"  Processed {i}/{len(data)} bytes ({i/len(data)*100:.1f}%)")

        print(
            f"RLE compression complete: {len(data)} → {len(compressed)} bytes")
        return bytes(compressed)

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress RLE-encoded data.

        Args:
            compressed_data: RLE-compressed bytes

        Returns:
            Original decompressed bytes
        """
        if not compressed_data or len(compressed_data) % 2 != 0:
            return b''

        decompressed = bytearray()

        for i in range(0, len(compressed_data), 2):
            count_byte = compressed_data[i]
            value_byte = compressed_data[i + 1]

            # Convert count (0 means 256)
            count = 256 if count_byte == 0 else count_byte

            # Repeat the byte 'count' times
            decompressed.extend([value_byte] * count)

        return bytes(decompressed)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using RLE.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(f"\n=== RLE Compression: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Analyze data characteristics
        unique_bytes = len(set(data))
        print(f"File analysis:")
        print(f"  Total bytes: {original_size}")
        print(f"  Unique bytes: {unique_bytes}")

        # Find longest run for analysis
        longest_run = 1
        current_run = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        print(f"  Longest run: {longest_run}")

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes) -> dict:
        """
        Analyze how well RLE will compress the data.

        Args:
            data: Input data to analyze

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        # Count runs of identical bytes
        runs = []
        current_run = 1

        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)  # Add final run

        # Calculate statistics
        total_runs = len(runs)
        runs_longer_than_1 = sum(1 for run in runs if run > 1)
        average_run_length = sum(runs) / len(runs)
        bytes_in_long_runs = sum(run for run in runs if run > 1)

        return {
            'total_runs': total_runs,
            'runs_with_repetition': runs_longer_than_1,
            'percentage_compressible': (runs_longer_than_1 / total_runs) * 100,
            'average_run_length': average_run_length,
            'bytes_in_runs': bytes_in_long_runs,
            'compression_potential': (bytes_in_long_runs / len(data)) * 100
        }


def compress_all_test_files():
    """Compress all files in the test directory using RLE."""
    compressor = RLECompressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/rle"

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        return []

    os.makedirs(output_dir, exist_ok=True)
    test_files = get_all_test_files(test_dir)
    results = []

    print(f"Found {len(test_files)} test files")
    print("=" * 60)

    for input_file in test_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{filename}.rle")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "RLE", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(
                f"  Runs with repetition: {analysis['runs_with_repetition']}/{analysis['total_runs']}")
            print(
                f"  Compression potential: {analysis['compression_potential']:.1f}%")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("Run-Length Encoding (RLE) Compression Demo")
    print("==========================================")
    print()
    print("RLE works by replacing runs of identical bytes with:")
    print("- Count of repetitions (1 byte)")
    print("- The repeated byte value (1 byte)")
    print()
    print("Best for: Files with many repeated bytes")
    print("Worst for: Files with no repetition (will expand!)")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("RLE COMPRESSION SUMMARY")
        print("="*60)

        total_original = sum(stat.original_size for stat in results)
        total_compressed = sum(stat.compressed_size for stat in results)
        overall_ratio = (
            (total_original - total_compressed) / total_original) * 100

        print(f"Files processed: {len(results)}")
        print(f"Total original size: {total_original:,} bytes")
        print(f"Total compressed size: {total_compressed:,} bytes")
        print(f"Overall compression ratio: {overall_ratio:.2f}%")

        # Find best and worst cases
        best = max(results, key=lambda x: x.compression_ratio)
        worst = min(results, key=lambda x: x.compression_ratio)

        print(
            f"\nBest compression: {best.compression_ratio:.1f}% on file with many repetitions")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% on file with little repetition")
