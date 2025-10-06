"""
Burrows-Wheeler Transform (BWT) with Move-to-Front and RLE

The Burrows-Wheeler Transform is a reversible text transformation that
rearranges data to make it more compressible. It's the foundation of
bzip2 compression.

The algorithm:
1. Create all rotations of the input string
2. Sort these rotations lexicographically  
3. Take the last column of the sorted matrix
4. Apply Move-to-Front encoding to reduce entropy
5. Apply Run-Length Encoding to compress runs

Example: "banana"
Rotations: banana, ananab, nanaba, anaban, nabana, abanan
Sorted: abanan, anaban, ananab, banana, nabana, nanaba
Last column: "nnbaaa" (this is more compressible!)

Used in: bzip2, some DNA sequence compression tools
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
from typing import Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BWTCompressor:
    """Burrows-Wheeler Transform compressor with Move-to-Front and RLE."""

    def __init__(self, block_size: int = 10000):
        """
        Initialize BWT compressor.

        Args:
            block_size: Size of blocks to process (larger = better compression, slower)
        """
        self.name = "BWT+MTF+RLE"
        self.block_size = block_size

    def burrows_wheeler_transform(self, data: bytes) -> Tuple[bytes, int]:
        """
        Apply Burrows-Wheeler Transform to data.

        Args:
            data: Input data block

        Returns:
            Tuple of (transformed data, original index)
        """
        if not data:
            return b'', 0

        print(f"  Applying BWT to {len(data)} bytes...")

        # Create all rotations
        rotations = []
        for i in range(len(data)):
            rotation = data[i:] + data[:i]
            rotations.append((rotation, i))

        # Sort rotations lexicographically
        rotations.sort(key=lambda x: x[0])

        # Extract last column and find original index
        last_column = bytearray()
        original_index = 0

        for i, (rotation, orig_pos) in enumerate(rotations):
            last_column.append(rotation[-1])
            if orig_pos == 0:  # This was the original string
                original_index = i

        print(f"    BWT complete, original index: {original_index}")
        return bytes(last_column), original_index

    def inverse_burrows_wheeler_transform(self, transformed_data: bytes, original_index: int) -> bytes:
        """
        Reverse Burrows-Wheeler Transform.

        Args:
            transformed_data: BWT-transformed data
            original_index: Index of original string in sorted matrix

        Returns:
            Original data
        """
        if not transformed_data:
            return b''

        n = len(transformed_data)

        # Create first column by sorting last column
        first_column = sorted(transformed_data)

        # Build transformation table
        counts = [0] * 256
        transform_table = [0] * n

        # Count occurrences and build table
        for i in range(n):
            char = transformed_data[i]
            transform_table[i] = counts[char]
            counts[char] += 1

        # Reconstruct original string
        result = bytearray()
        index = original_index

        for _ in range(n):
            char = first_column[index]
            result.append(char)

            # Find next index
            rank = transform_table[index]
            index = 0
            count = 0
            for i in range(n):
                if transformed_data[i] == char:
                    if count == rank:
                        index = i
                        break
                    count += 1

        return bytes(result)

    def move_to_front_encode(self, data: bytes) -> bytes:
        """
        Apply Move-to-Front encoding.

        MTF maintains a list of symbols. For each symbol:
        1. Output its current position in the list
        2. Move it to the front of the list

        This tends to create many small numbers (especially 0s) after BWT.

        Args:
            data: Input data

        Returns:
            MTF-encoded data
        """
        if not data:
            return b''

        print(f"    Applying Move-to-Front encoding...")

        # Initialize symbol list (0-255)
        symbols = list(range(256))
        result = bytearray()

        for byte in data:
            # Find position of byte in list
            position = symbols.index(byte)
            result.append(position)

            # Move byte to front
            symbols.pop(position)
            symbols.insert(0, byte)

        # Count zeros for analysis
        zero_count = result.count(0)
        print(
            f"    MTF: {zero_count}/{len(result)} zeros ({zero_count/len(result)*100:.1f}%)")

        return bytes(result)

    def move_to_front_decode(self, encoded_data: bytes) -> bytes:
        """
        Reverse Move-to-Front encoding.

        Args:
            encoded_data: MTF-encoded data

        Returns:
            Original data
        """
        if not encoded_data:
            return b''

        # Initialize symbol list
        symbols = list(range(256))
        result = bytearray()

        for position in encoded_data:
            # Get symbol at position
            symbol = symbols[position]
            result.append(symbol)

            # Move symbol to front
            symbols.pop(position)
            symbols.insert(0, symbol)

        return bytes(result)

    def run_length_encode(self, data: bytes) -> bytes:
        """
        Apply simple Run-Length Encoding optimized for zeros.

        Format:
        - 0 followed by count byte: run of zeros
        - Non-zero byte: literal value

        Args:
            data: Input data

        Returns:
            RLE-encoded data
        """
        if not data:
            return b''

        print(f"    Applying RLE...")

        result = bytearray()
        i = 0

        while i < len(data):
            if data[i] == 0:
                # Count consecutive zeros
                count = 0
                while i < len(data) and data[i] == 0 and count < 255:
                    count += 1
                    i += 1

                # Encode run of zeros
                result.extend([0, count])
            else:
                # Literal byte
                result.append(data[i])
                i += 1

        zero_runs = sum(1 for i in range(
            0, len(result)-1, 2) if result[i] == 0)
        print(f"    RLE: encoded {zero_runs} zero runs")

        return bytes(result)

    def run_length_decode(self, encoded_data: bytes) -> bytes:
        """
        Reverse Run-Length Encoding.

        Args:
            encoded_data: RLE-encoded data

        Returns:
            Original data
        """
        if not encoded_data:
            return b''

        result = bytearray()
        i = 0

        while i < len(encoded_data):
            if encoded_data[i] == 0 and i + 1 < len(encoded_data):
                # Run of zeros
                count = encoded_data[i + 1]
                result.extend([0] * count)
                i += 2
            else:
                # Literal byte
                result.append(encoded_data[i])
                i += 1

        return bytes(result)

    def compress_block(self, block: bytes) -> bytes:
        """
        Compress a single block using BWT + MTF + RLE.

        Args:
            block: Input block

        Returns:
            Compressed block with header
        """
        print(f"  Compressing block of {len(block)} bytes...")

        # Apply BWT
        bwt_data, original_index = self.burrows_wheeler_transform(block)

        # Apply Move-to-Front
        mtf_data = self.move_to_front_encode(bwt_data)

        # Apply Run-Length Encoding
        rle_data = self.run_length_encode(mtf_data)

        # Create compressed block with header
        result = bytearray()
        result.extend(len(block).to_bytes(4, 'big'))      # Original block size
        result.extend(original_index.to_bytes(4, 'big'))  # BWT original index
        result.extend(len(rle_data).to_bytes(4, 'big'))   # Compressed size
        result.extend(rle_data)                           # Compressed data

        print(f"  Block compressed: {len(block)} → {len(result)} bytes")
        return bytes(result)

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using BWT in blocks.

        Args:
            data: Input data

        Returns:
            Compressed data
        """
        if not data:
            return b''

        print(f"Compressing {len(data)} bytes using BWT+MTF+RLE...")
        print(f"Block size: {self.block_size}")

        # Split data into blocks
        blocks = []
        for i in range(0, len(data), self.block_size):
            block = data[i:i + self.block_size]
            blocks.append(block)

        print(f"Split into {len(blocks)} blocks")

        # Compress each block
        compressed_blocks = []
        for i, block in enumerate(blocks):
            print(f"Processing block {i+1}/{len(blocks)}...")
            compressed_block = self.compress_block(block)
            compressed_blocks.append(compressed_block)

        # Combine compressed blocks with header
        result = bytearray()
        result.extend(len(blocks).to_bytes(4, 'big'))  # Number of blocks

        for compressed_block in compressed_blocks:
            result.extend(compressed_block)

        print(f"BWT compression complete: {len(data)} → {len(result)} bytes")
        return bytes(result)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using BWT.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(
            f"\n=== BWT+MTF+RLE Compression: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Analyze data characteristics
        print(f"File analysis:")
        print(f"  Total bytes: {original_size}")

        if data:
            # Count byte frequencies for analysis
            unique_bytes = len(set(data))

            # Sample data to estimate sortability
            sample_size = min(1000, len(data))
            sample = data[:sample_size]

            # Count repeated substrings in sample
            repeated_patterns = 0
            for i in range(len(sample) - 2):
                pattern = sample[i:i+3]
                if pattern in sample[i+3:]:
                    repeated_patterns += 1

            pattern_density = (repeated_patterns /
                               len(sample)) * 100 if sample else 0

            print(f"  Unique bytes: {unique_bytes}")
            print(f"  Pattern repetition: {pattern_density:.1f}%")

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes, sample_size: int = 5000) -> dict:
        """
        Analyze how well BWT will compress the data.

        Args:
            data: Input data to analyze
            sample_size: Size of sample to analyze

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        # Analyze sample for speed
        sample_end = min(sample_size, len(data))
        sample = data[:sample_end]

        # Count character repetition after a simple transform simulation
        char_counts = {}
        for byte in sample:
            char_counts[byte] = char_counts.get(byte, 0) + 1

        # Estimate clustering potential (how much BWT might help)
        max_char_freq = max(char_counts.values()) if char_counts else 0
        clustering_potential = (
            max_char_freq / len(sample)) * 100 if sample else 0

        # Count runs in original data
        runs = 1
        for i in range(1, len(sample)):
            if sample[i] != sample[i-1]:
                runs += 1

        run_length_avg = len(sample) / runs if runs > 0 else 0

        return {
            'sample_size': len(sample),
            'unique_chars': len(char_counts),
            'clustering_potential': clustering_potential,
            'original_runs': runs,
            'average_run_length': run_length_avg,
            # Rough estimate
            'compression_estimate': min(clustering_potential * 1.5, 80)
        }


def compress_all_test_files():
    """Compress all files in the test directory using BWT."""
    compressor = BWTCompressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/bwt"

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
        output_file = os.path.join(output_dir, f"{filename}.bwt")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "BWT+MTF+RLE", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(
                f"  Clustering potential: {analysis.get('clustering_potential', 0):.1f}%")
            print(f"  Original runs: {analysis.get('original_runs', 0)}")
            print(
                f"  Compression estimate: {analysis.get('compression_estimate', 0):.1f}%")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("Burrows-Wheeler Transform + Move-to-Front + RLE Demo")
    print("===================================================")
    print()
    print("BWT rearranges data to create clusters of similar characters:")
    print("- Creates all rotations of input")
    print("- Sorts rotations lexicographically")
    print("- Takes last column (clustered data)")
    print("- Applies Move-to-Front encoding")
    print("- Applies Run-Length Encoding")
    print()
    print("Best for: Text files, repetitive data")
    print("Used in: bzip2 compression algorithm")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("BWT+MTF+RLE COMPRESSION SUMMARY")
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
            f"\nBest compression: {best.compression_ratio:.1f}% on text with repetitive patterns")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% on random or already compressed data")
