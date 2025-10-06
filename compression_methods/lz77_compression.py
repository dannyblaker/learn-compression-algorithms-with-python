"""
LZ77 Compression Algorithm

LZ77 is a dictionary-based compression algorithm that uses a sliding window
to find repeated substrings. It replaces repeated data with references to
previous occurrences.

The algorithm maintains:
- Search buffer: Previously seen data (lookback window)
- Lookahead buffer: Data to be compressed next

For each position, it finds the longest match in the search buffer and
outputs a tuple: (distance back, length of match, next literal character)

This is the foundation for many modern compression algorithms like gzip.
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
from typing import Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LZ77Compressor:
    """LZ77 compression implementation with sliding window."""

    def __init__(self, search_buffer_size: int = 1024, lookahead_buffer_size: int = 18):
        """
        Initialize LZ77 compressor.

        Args:
            search_buffer_size: Size of search buffer (lookback window)
            lookahead_buffer_size: Size of lookahead buffer
        """
        self.name = "LZ77"
        self.search_buffer_size = search_buffer_size
        self.lookahead_buffer_size = lookahead_buffer_size
        self.min_match_length = 3  # Minimum match length to be worthwhile

    def find_longest_match(self, data: bytes, current_pos: int) -> Tuple[int, int]:
        """
        Find the longest match in the search buffer.

        Args:
            data: Input data
            current_pos: Current position in data

        Returns:
            Tuple of (distance back, match length)
        """
        # Define search boundaries
        search_start = max(0, current_pos - self.search_buffer_size)
        search_end = current_pos
        lookahead_end = min(len(data), current_pos +
                            self.lookahead_buffer_size)

        best_distance = 0
        best_length = 0

        # Search for matches in the search buffer
        for search_pos in range(search_start, search_end):
            match_length = 0

            # Extend match as far as possible
            while (current_pos + match_length < lookahead_end and
                   search_pos + match_length < current_pos and
                   data[search_pos + match_length] == data[current_pos + match_length]):
                match_length += 1

            # Update best match if this one is longer
            if match_length >= self.min_match_length and match_length > best_length:
                best_distance = current_pos - search_pos
                best_length = match_length

        return best_distance, best_length

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using LZ77 algorithm.

        Output format for each token:
        - Type flag (1 byte): 0 for literal, 1 for match
        - For literal: byte value (1 byte)
        - For match: distance (2 bytes) + length (2 bytes)

        Args:
            data: Input bytes to compress

        Returns:
            Compressed bytes
        """
        if not data:
            return b''

        print(f"Compressing {len(data)} bytes using LZ77...")
        print(f"Search buffer size: {self.search_buffer_size}")
        print(f"Lookahead buffer size: {self.lookahead_buffer_size}")

        compressed = bytearray()
        pos = 0
        match_count = 0
        literal_count = 0

        while pos < len(data):
            # Find longest match
            distance, length = self.find_longest_match(data, pos)

            if length >= self.min_match_length:
                # Output match token
                compressed.append(1)  # Match flag
                compressed.extend(distance.to_bytes(
                    2, 'big'))  # Distance (2 bytes)
                compressed.extend(length.to_bytes(
                    2, 'big'))    # Length (2 bytes)

                pos += length
                match_count += 1
            else:
                # Output literal token
                compressed.append(0)  # Literal flag
                compressed.append(data[pos])  # Literal byte

                pos += 1
                literal_count += 1

            # Progress indicator
            if pos % 5000 == 0:
                print(
                    f"  Processed {pos}/{len(data)} bytes ({pos/len(data)*100:.1f}%)")

        print(f"LZ77 compression complete:")
        print(f"  Matches found: {match_count}")
        print(f"  Literals: {literal_count}")
        print(f"  Output size: {len(data)} → {len(compressed)} bytes")

        return bytes(compressed)

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress LZ77-encoded data.

        Args:
            compressed_data: LZ77-compressed bytes

        Returns:
            Original decompressed bytes
        """
        decompressed = bytearray()
        pos = 0

        while pos < len(compressed_data):
            if pos >= len(compressed_data):
                break

            flag = compressed_data[pos]
            pos += 1

            if flag == 0:  # Literal
                if pos >= len(compressed_data):
                    break
                decompressed.append(compressed_data[pos])
                pos += 1
            else:  # Match
                if pos + 3 >= len(compressed_data):
                    break

                distance = int.from_bytes(compressed_data[pos:pos+2], 'big')
                length = int.from_bytes(compressed_data[pos+2:pos+4], 'big')
                pos += 4

                # Copy from previous data
                start_pos = len(decompressed) - distance
                for i in range(length):
                    if start_pos + i >= 0 and start_pos + i < len(decompressed):
                        decompressed.append(decompressed[start_pos + i])

        return bytes(decompressed)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using LZ77.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(
            f"\n=== LZ77 Compression: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Analyze data for repetition patterns
        print(f"File analysis:")
        print(f"  Total bytes: {original_size}")

        # Quick analysis: count potential matches
        sample_size = min(10000, len(data))
        sample_matches = 0
        for i in range(sample_size - 10):
            substring = data[i:i+3]
            if substring in data[i+3:i+100]:  # Look ahead for same pattern
                sample_matches += 1

        match_density = (sample_matches / sample_size) * \
            100 if sample_size > 0 else 0
        print(f"  Estimated match density: {match_density:.1f}%")

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes, sample_size: int = 10000) -> dict:
        """
        Analyze how well LZ77 will compress the data.

        Args:
            data: Input data to analyze
            sample_size: Number of bytes to sample for analysis

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        # Sample data for analysis to avoid long processing times
        sample_end = min(sample_size, len(data))
        sample_data = data[:sample_end]

        total_matches = 0
        total_match_length = 0
        match_lengths = []

        # Analyze matches in sample
        for pos in range(len(sample_data)):
            distance, length = self.find_longest_match(data, pos)
            if length >= self.min_match_length:
                total_matches += 1
                total_match_length += length
                match_lengths.append(length)

        avg_match_length = total_match_length / \
            total_matches if total_matches > 0 else 0
        compression_estimate = (total_match_length / len(sample_data)) * 100

        return {
            'matches_found': total_matches,
            'average_match_length': avg_match_length,
            'longest_match': max(match_lengths) if match_lengths else 0,
            'compression_potential': compression_estimate,
            'match_density': (total_matches / len(sample_data)) * 100
        }


def compress_all_test_files():
    """Compress all files in the test directory using LZ77."""
    compressor = LZ77Compressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/lz77"

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
        output_file = os.path.join(output_dir, f"{filename}.lz77")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "LZ77", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(f"  Matches found: {analysis['matches_found']}")
            print(
                f"  Average match length: {analysis['average_match_length']:.1f}")
            print(
                f"  Compression potential: {analysis['compression_potential']:.1f}%")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("LZ77 Compression Demo")
    print("====================")
    print()
    print("LZ77 uses a sliding window to find repeated substrings:")
    print("- Search buffer: looks back for matches")
    print("- Lookahead buffer: data to be compressed")
    print("- Outputs: (distance, length, next_char) tuples")
    print()
    print("Best for: Files with repeated patterns or sequences")
    print("Foundation for: gzip, PNG compression, and many others")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("LZ77 COMPRESSION SUMMARY")
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
            f"\nBest compression: {best.compression_ratio:.1f}% on file with repetitive patterns")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% on file with little repetition")
