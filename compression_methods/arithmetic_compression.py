"""
Arithmetic Coding Compression Algorithm

Arithmetic coding is an entropy encoding technique that represents an entire
message as a single number in the range [0, 1). It's theoretically optimal
and can achieve better compression than Huffman coding.

The algorithm:
1. Build probability model of input symbols
2. Assign each symbol a range proportional to its probability
3. Encode message by successively narrowing the range
4. Output the final range as a single number

Example: For message "HELLO" with probabilities:
H: 0.2, E: 0.1, L: 0.4, O: 0.3
Start with range [0, 1), narrow for each character.

This implementation uses fixed-point arithmetic to avoid floating-point
precision issues.
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
from collections import Counter
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ArithmeticCompressor:
    """Arithmetic coding compressor with fixed-point arithmetic."""

    def __init__(self, precision: int = 32):
        """
        Initialize arithmetic compressor.

        Args:
            precision: Number of bits for fixed-point arithmetic
        """
        self.name = "Arithmetic"
        self.precision = precision
        self.max_range = 2 ** precision - 1
        self.quarter = self.max_range // 4
        self.half = 2 * self.quarter
        self.three_quarters = 3 * self.quarter

    def build_probability_model(self, data: bytes) -> Tuple[Dict[int, int], int]:
        """
        Build cumulative frequency model for arithmetic coding.

        Args:
            data: Input data

        Returns:
            Tuple of (cumulative frequencies, total count)
        """
        print("Building probability model...")

        # Count frequencies
        frequencies = Counter(data)
        total_count = len(data)

        # Build cumulative frequency table
        cumulative_freq = {}
        cumulative_count = 0

        # Sort by byte value for consistency
        for byte_val in sorted(frequencies.keys()):
            cumulative_freq[byte_val] = cumulative_count
            cumulative_count += frequencies[byte_val]

        # Add end-of-file marker
        eof_symbol = 256
        cumulative_freq[eof_symbol] = cumulative_count
        total_count += 1  # Add EOF to total

        print(f"Probability model built:")
        print(f"  Unique symbols: {len(frequencies)}")
        print(f"  Total symbols: {total_count}")

        # Show most/least frequent for analysis
        sorted_freqs = sorted(frequencies.items(),
                              key=lambda x: x[1], reverse=True)
        print(
            f"  Most frequent: byte {sorted_freqs[0][0]} ({sorted_freqs[0][1]} times)")
        print(
            f"  Least frequent: byte {sorted_freqs[-1][0]} ({sorted_freqs[-1][1]} times)")

        return cumulative_freq, total_count

    def encode_symbol(self, symbol: int, low: int, high: int,
                      cumulative_freq: Dict[int, int], total_count: int) -> Tuple[int, int]:
        """
        Encode a single symbol by narrowing the range.

        Args:
            symbol: Symbol to encode
            low: Current low range
            high: Current high range
            cumulative_freq: Cumulative frequency table
            total_count: Total symbol count

        Returns:
            New (low, high) range
        """
        range_size = high - low + 1

        # Get symbol's frequency range
        symbol_low = cumulative_freq[symbol]

        # Find symbol's high frequency
        sorted_symbols = sorted(cumulative_freq.keys())
        symbol_idx = sorted_symbols.index(symbol)

        if symbol_idx < len(sorted_symbols) - 1:
            next_symbol = sorted_symbols[symbol_idx + 1]
            symbol_high = cumulative_freq[next_symbol] - 1
        else:
            symbol_high = total_count - 1

        # Calculate new range
        new_high = low + (range_size * (symbol_high + 1)) // total_count - 1
        new_low = low + (range_size * symbol_low) // total_count

        return new_low, new_high

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using arithmetic coding.

        Args:
            data: Input bytes to compress

        Returns:
            Compressed bytes
        """
        if not data:
            return b''

        print(f"Compressing {len(data)} bytes using Arithmetic Coding...")

        # Build probability model
        cumulative_freq, total_count = self.build_probability_model(data)

        # Initialize range
        low = 0
        high = self.max_range
        pending_bits = 0
        output_bits = []

        # Encode each symbol
        symbols_to_encode = list(data) + [256]  # Add EOF marker

        for i, symbol in enumerate(symbols_to_encode):
            # Encode symbol
            low, high = self.encode_symbol(
                symbol, low, high, cumulative_freq, total_count)

            # Handle precision issues with bit output
            while True:
                if high < self.half:
                    # Output 0
                    output_bits.append(0)
                    for _ in range(pending_bits):
                        output_bits.append(1)
                    pending_bits = 0
                elif low >= self.half:
                    # Output 1
                    output_bits.append(1)
                    for _ in range(pending_bits):
                        output_bits.append(0)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarters:
                    # Scale
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                # Scale ranges
                low = 2 * low
                high = 2 * high + 1

                if high > self.max_range:
                    high = self.max_range

            # Progress indicator
            if i % 1000 == 0 and i > 0:
                print(
                    f"  Encoded {i}/{len(symbols_to_encode)} symbols ({i/len(symbols_to_encode)*100:.1f}%)")

        # Output final bits
        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            for _ in range(pending_bits):
                output_bits.append(1)
        else:
            output_bits.append(1)
            for _ in range(pending_bits):
                output_bits.append(0)

        # Convert bits to bytes
        compressed_data = self.bits_to_bytes(
            output_bits, cumulative_freq, total_count)

        print(f"Arithmetic coding complete:")
        print(f"  Output bits: {len(output_bits)}")
        print(f"  Compressed size: {len(data)} → {len(compressed_data)} bytes")

        return compressed_data

    def bits_to_bytes(self, bits: list, cumulative_freq: Dict[int, int], total_count: int) -> bytes:
        """
        Convert bit sequence to bytes with header information.

        Args:
            bits: List of bits (0s and 1s)
            cumulative_freq: Frequency table for decompression
            total_count: Total symbol count

        Returns:
            Compressed data with header
        """
        result = bytearray()

        # Store frequency table size
        result.extend(len(cumulative_freq).to_bytes(2, 'big'))

        # Store total count
        result.extend(total_count.to_bytes(4, 'big'))

        # Store frequency table
        for symbol in sorted(cumulative_freq.keys()):
            # Symbol (can be > 255 for EOF)
            result.extend(symbol.to_bytes(2, 'big'))
            result.extend(cumulative_freq[symbol].to_bytes(
                4, 'big'))  # Cumulative frequency

        # Store number of bits
        result.extend(len(bits).to_bytes(4, 'big'))

        # Pack bits into bytes
        byte_val = 0
        bit_count = 0

        for bit in bits:
            byte_val = (byte_val << 1) | bit
            bit_count += 1

            if bit_count == 8:
                result.append(byte_val)
                byte_val = 0
                bit_count = 0

        # Handle remaining bits
        if bit_count > 0:
            byte_val = byte_val << (8 - bit_count)
            result.append(byte_val)

        return bytes(result)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using arithmetic coding.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(
            f"\n=== Arithmetic Coding: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Analyze data entropy
        print(f"File analysis:")
        print(f"  Total bytes: {original_size}")

        if data:
            frequencies = Counter(data)
            unique_bytes = len(frequencies)

            # Calculate entropy
            entropy = 0
            for freq in frequencies.values():
                probability = freq / len(data)
                if probability > 0:
                    entropy -= probability * (probability.bit_length() - 1)

            print(f"  Unique bytes: {unique_bytes}")
            print(f"  Entropy: {entropy:.2f} bits per byte")
            print(
                f"  Theoretical compression: {((8 - entropy) / 8) * 100:.1f}%")

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes) -> dict:
        """
        Analyze entropy and theoretical compression potential.

        Args:
            data: Input data to analyze

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        frequencies = Counter(data)
        total_bytes = len(data)

        # Calculate entropy (theoretical compression limit)
        entropy = 0
        for freq in frequencies.values():
            probability = freq / total_bytes
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)

        # Calculate compression potential
        bits_per_byte = entropy
        compression_ratio = ((8 - bits_per_byte) / 8) * 100

        # Find distribution characteristics
        max_freq = max(frequencies.values())
        min_freq = min(frequencies.values())

        return {
            'unique_symbols': len(frequencies),
            'entropy_bits_per_byte': entropy,
            'theoretical_compression_ratio': compression_ratio,
            'max_frequency': max_freq,
            'min_frequency': min_freq,
            'frequency_ratio': max_freq / min_freq if min_freq > 0 else float('inf')
        }


def compress_all_test_files():
    """Compress all files in the test directory using Arithmetic Coding."""
    compressor = ArithmeticCompressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/arithmetic"

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
        output_file = os.path.join(output_dir, f"{filename}.arith")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "Arithmetic", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(
                f"  Entropy: {analysis.get('entropy_bits_per_byte', 0):.2f} bits/byte")
            print(
                f"  Theoretical best: {analysis.get('theoretical_compression_ratio', 0):.1f}%")
            print(
                f"  Frequency ratio: {analysis.get('frequency_ratio', 0):.1f}:1")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("Arithmetic Coding Compression Demo")
    print("=================================")
    print()
    print("Arithmetic coding represents entire messages as single numbers:")
    print("- Assigns probability ranges to each symbol")
    print("- Narrows range for each symbol in message")
    print("- Theoretically optimal compression")
    print()
    print("Best for: Any data (optimal for given probability model)")
    print("Advantage: Can achieve fractional bits per symbol")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("ARITHMETIC CODING SUMMARY")
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
            f"\nBest compression: {best.compression_ratio:.1f}% (close to theoretical optimum)")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% (high entropy data)")
