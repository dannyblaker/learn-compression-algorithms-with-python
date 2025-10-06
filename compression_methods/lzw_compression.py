"""
LZW (Lempel-Ziv-Welch) Compression Algorithm

LZW is an adaptive dictionary compression algorithm. Unlike LZ77, it builds
a dictionary of patterns dynamically as it processes the data. It starts
with all single bytes in the dictionary, then adds new patterns as they
are encountered.

The algorithm:
1. Initialize dictionary with all possible single bytes (0-255)
2. Read longest string that's already in dictionary
3. Output the code for that string
4. Add the string + next character to dictionary
5. Repeat from step 2

Used in: GIF files, Unix compress utility, some PDF compression
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LZWCompressor:
    """LZW compression implementation with adaptive dictionary."""

    def __init__(self, max_code_size: int = 16):
        """
        Initialize LZW compressor.

        Args:
            max_code_size: Maximum size of codes in bits (limits dictionary size)
        """
        self.name = "LZW"
        self.max_code_size = max_code_size
        self.max_dict_size = 2 ** max_code_size - 1

    def initialize_dictionary(self) -> Dict[bytes, int]:
        """
        Initialize dictionary with all single-byte values.

        Returns:
            Dictionary mapping byte sequences to codes
        """
        dictionary = {}
        for i in range(256):
            dictionary[bytes([i])] = i
        return dictionary

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using LZW algorithm.

        Args:
            data: Input bytes to compress

        Returns:
            Compressed bytes (sequence of variable-length codes)
        """
        if not data:
            return b''

        print(f"Compressing {len(data)} bytes using LZW...")
        print(f"Maximum dictionary size: {self.max_dict_size}")

        # Initialize dictionary and variables
        dictionary = self.initialize_dictionary()
        next_code = 256
        current_string = bytes([data[0]])
        codes = []

        dictionary_hits = 0
        dictionary_misses = 0

        for i in range(1, len(data)):
            next_char = bytes([data[i]])
            extended_string = current_string + next_char

            if extended_string in dictionary:
                # String is in dictionary, extend current string
                current_string = extended_string
                dictionary_hits += 1
            else:
                # String not in dictionary
                codes.append(dictionary[current_string])
                dictionary_misses += 1

                # Add new string to dictionary if there's space
                if next_code <= self.max_dict_size:
                    dictionary[extended_string] = next_code
                    next_code += 1

                current_string = next_char

            # Progress indicator
            if i % 10000 == 0:
                print(
                    f"  Processed {i}/{len(data)} bytes ({i/len(data)*100:.1f}%)")
                print(f"    Dictionary size: {len(dictionary)}")

        # Don't forget the last string
        codes.append(dictionary[current_string])

        print(f"LZW compression analysis:")
        print(f"  Dictionary final size: {len(dictionary)}")
        print(f"  Dictionary hits: {dictionary_hits}")
        print(f"  Dictionary misses: {dictionary_misses}")
        print(f"  Output codes: {len(codes)}")

        # Convert codes to bytes
        compressed_data = self.codes_to_bytes(codes)

        print(
            f"LZW compression complete: {len(data)} → {len(compressed_data)} bytes")
        return compressed_data

    def codes_to_bytes(self, codes: List[int]) -> bytes:
        """
        Convert list of codes to packed bytes.

        Uses variable-length encoding:
        - Codes 0-255: 1 byte
        - Codes 256-65535: 2 bytes with continuation bit

        Args:
            codes: List of LZW codes

        Returns:
            Packed bytes
        """
        result = bytearray()

        for code in codes:
            if code <= 255:
                # Single byte code
                result.append(code)
            elif code <= 65535:
                # Two byte code with high bit set in first byte
                # High byte with continuation bit
                result.append(0x80 | (code >> 8))
                result.append(code & 0xFF)         # Low byte
            else:
                # Code too large - this shouldn't happen with our max_code_size
                raise ValueError(f"Code {code} too large for encoding")

        return bytes(result)

    def bytes_to_codes(self, data: bytes) -> List[int]:
        """
        Convert packed bytes back to list of codes.

        Args:
            data: Packed bytes

        Returns:
            List of LZW codes
        """
        codes = []
        i = 0

        while i < len(data):
            if data[i] & 0x80:  # High bit set - two byte code
                if i + 1 >= len(data):
                    break
                high_byte = data[i] & 0x7F  # Remove continuation bit
                low_byte = data[i + 1]
                code = (high_byte << 8) | low_byte
                codes.append(code)
                i += 2
            else:  # Single byte code
                codes.append(data[i])
                i += 1

        return codes

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress LZW-encoded data.

        Args:
            compressed_data: LZW-compressed bytes

        Returns:
            Original decompressed bytes
        """
        if not compressed_data:
            return b''

        # Convert bytes to codes
        codes = self.bytes_to_codes(compressed_data)

        if not codes:
            return b''

        # Initialize dictionary for decompression (reverse mapping)
        dictionary = {}
        for i in range(256):
            dictionary[i] = bytes([i])

        next_code = 256
        result = bytearray()

        # First code
        old_code = codes[0]
        result.extend(dictionary[old_code])

        for code in codes[1:]:
            if code in dictionary:
                string = dictionary[code]
            elif code == next_code:
                # Special case: code not yet in dictionary
                string = dictionary[old_code] + dictionary[old_code][:1]
            else:
                raise ValueError(f"Invalid LZW code: {code}")

            result.extend(string)

            # Add new entry to dictionary
            if next_code <= self.max_dict_size:
                dictionary[next_code] = dictionary[old_code] + string[:1]
                next_code += 1

            old_code = code

        return bytes(result)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using LZW.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(f"\n=== LZW Compression: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Analyze data patterns
        print(f"File analysis:")
        print(f"  Total bytes: {original_size}")

        # Count unique byte patterns of length 2
        if len(data) > 1:
            patterns = set()
            for i in range(len(data) - 1):
                patterns.add(data[i:i+2])
            print(f"  Unique 2-byte patterns: {len(patterns)}")

        # Count unique byte patterns of length 3
        if len(data) > 2:
            patterns3 = set()
            for i in range(min(1000, len(data) - 2)):  # Sample to avoid long processing
                patterns3.add(data[i:i+3])
            print(f"  Unique 3-byte patterns (sample): {len(patterns3)}")

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes, sample_size: int = 5000) -> dict:
        """
        Analyze how well LZW will compress the data.

        Args:
            data: Input data to analyze
            sample_size: Number of bytes to sample for analysis

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        # Sample data for analysis
        sample_end = min(sample_size, len(data))
        sample_data = data[:sample_end]

        # Count repeated patterns of various lengths
        pattern_counts = {}
        for pattern_length in [2, 3, 4]:
            patterns = {}
            for i in range(len(sample_data) - pattern_length + 1):
                pattern = sample_data[i:i+pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1

            repeated_patterns = sum(
                1 for count in patterns.values() if count > 1)
            pattern_counts[f'repeated_{pattern_length}_byte_patterns'] = repeated_patterns
            pattern_counts[f'total_{pattern_length}_byte_patterns'] = len(
                patterns)

        # Estimate compression potential based on pattern repetition
        total_2_byte = pattern_counts.get('total_2_byte_patterns', 1)
        repeated_2_byte = pattern_counts.get('repeated_2_byte_patterns', 0)
        compression_estimate = (repeated_2_byte / total_2_byte) * 100

        return {
            'sample_size': len(sample_data),
            **pattern_counts,
            'compression_potential': compression_estimate
        }


def compress_all_test_files():
    """Compress all files in the test directory using LZW."""
    compressor = LZWCompressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/lzw"

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
        output_file = os.path.join(output_dir, f"{filename}.lzw")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "LZW", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(
                f"  Repeated 2-byte patterns: {analysis.get('repeated_2_byte_patterns', 0)}")
            print(
                f"  Repeated 3-byte patterns: {analysis.get('repeated_3_byte_patterns', 0)}")
            print(
                f"  Compression potential: {analysis.get('compression_potential', 0):.1f}%")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("LZW (Lempel-Ziv-Welch) Compression Demo")
    print("======================================")
    print()
    print("LZW builds an adaptive dictionary as it processes data:")
    print("- Starts with all single bytes (0-255)")
    print("- Adds new patterns dynamically")
    print("- Outputs codes instead of original bytes")
    print()
    print("Best for: Files with repeated patterns that grow over time")
    print("Used in: GIF compression, Unix compress utility")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("LZW COMPRESSION SUMMARY")
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
            f"\nBest compression: {best.compression_ratio:.1f}% on file with adaptive patterns")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% on file with random data")
