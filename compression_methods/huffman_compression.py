"""
Huffman Coding Compression

Huffman coding is a lossless compression algorithm that uses variable-length
prefix codes based on the frequency of characters. Characters that appear
more frequently get shorter codes.

The algorithm:
1. Count frequency of each byte in the input
2. Build a binary tree (Huffman tree) based on frequencies
3. Assign binary codes: left=0, right=1
4. Replace each byte with its corresponding code
5. Store the tree structure for decompression

Example: If 'A' appears frequently, it might get code '0'
         If 'Z' appears rarely, it might get code '111010'
"""

from utils.file_utils import (
    get_file_size, read_file_binary, write_file_binary,
    get_all_test_files, CompressionStats, print_compression_stats, measure_compression_time
)
import os
import sys
import heapq
from collections import defaultdict, Counter
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HuffmanNode:
    """Node in the Huffman tree."""

    def __init__(self, char: Optional[int] = None, freq: int = 0,
                 left: Optional['HuffmanNode'] = None, right: Optional['HuffmanNode'] = None):
        self.char = char  # Byte value (0-255) or None for internal nodes
        self.freq = freq  # Frequency of this character/subtree
        self.left = left
        self.right = right

    def __lt__(self, other):
        """For heapq comparison."""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.char is not None


class HuffmanCompressor:
    """Huffman coding compressor implementation."""

    def __init__(self):
        self.name = "Huffman"
        self.codes = {}  # Mapping from byte value to binary code string
        self.tree = None

    def build_frequency_table(self, data: bytes) -> Dict[int, int]:
        """
        Count frequency of each byte in the data.

        Args:
            data: Input bytes

        Returns:
            Dictionary mapping byte values to frequencies
        """
        print("Building frequency table...")
        frequencies = Counter(data)

        print(f"Found {len(frequencies)} unique bytes")

        # Show most and least frequent bytes for analysis
        sorted_freqs = sorted(frequencies.items(),
                              key=lambda x: x[1], reverse=True)
        print(
            f"Most frequent byte: {sorted_freqs[0][0]} (appears {sorted_freqs[0][1]} times)")
        print(
            f"Least frequent byte: {sorted_freqs[-1][0]} (appears {sorted_freqs[-1][1]} times)")

        return frequencies

    def build_huffman_tree(self, frequencies: Dict[int, int]) -> HuffmanNode:
        """
        Build Huffman tree from frequency table.

        Args:
            frequencies: Byte frequencies

        Returns:
            Root node of Huffman tree
        """
        print("Building Huffman tree...")

        if not frequencies:
            return None

        # Handle single character case
        if len(frequencies) == 1:
            char, freq = next(iter(frequencies.items()))
            return HuffmanNode(char=char, freq=freq)

        # Create heap of nodes
        heap = []
        for char, freq in frequencies.items():
            node = HuffmanNode(char=char, freq=freq)
            heapq.heappush(heap, node)

        print(f"Starting with {len(heap)} leaf nodes")

        # Build tree by repeatedly combining least frequent nodes
        while len(heap) > 1:
            # Take two nodes with smallest frequencies
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            # Create internal node
            merged = HuffmanNode(
                freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)

        tree_root = heap[0]
        print("Huffman tree construction complete")
        return tree_root

    def generate_codes(self, root: HuffmanNode) -> Dict[int, str]:
        """
        Generate binary codes from Huffman tree.

        Args:
            root: Root of Huffman tree

        Returns:
            Dictionary mapping byte values to binary code strings
        """
        if not root:
            return {}

        codes = {}

        def traverse(node: HuffmanNode, code: str = ""):
            if node.is_leaf():
                # Leaf node: assign code (or "0" for single character)
                codes[node.char] = code if code else "0"
            else:
                # Internal node: traverse children
                if node.left:
                    traverse(node.left, code + "0")
                if node.right:
                    traverse(node.right, code + "1")

        traverse(root)

        print(f"Generated {len(codes)} Huffman codes")

        # Show some example codes for analysis
        sorted_codes = sorted(codes.items(), key=lambda x: len(x[1]))
        print(
            f"Shortest code: byte {sorted_codes[0][0]} = '{sorted_codes[0][1]}'")
        print(
            f"Longest code: byte {sorted_codes[-1][0]} = '{sorted_codes[-1][1]}'")

        return codes

    def encode_data(self, data: bytes, codes: Dict[int, str]) -> str:
        """
        Encode data using Huffman codes.

        Args:
            data: Input bytes
            codes: Huffman codes dictionary

        Returns:
            Binary string of encoded data
        """
        print(f"Encoding {len(data)} bytes...")

        encoded_bits = []
        for i, byte in enumerate(data):
            encoded_bits.append(codes[byte])

            # Progress indicator
            if i % 10000 == 0 and i > 0:
                print(
                    f"  Encoded {i}/{len(data)} bytes ({i/len(data)*100:.1f}%)")

        encoded_string = ''.join(encoded_bits)
        print(
            f"Encoding complete: {len(data)*8} bits → {len(encoded_string)} bits")

        return encoded_string

    def serialize_tree(self, root: HuffmanNode) -> bytes:
        """
        Serialize Huffman tree for storage.

        Uses pre-order traversal:
        - Internal node: 0
        - Leaf node: 1 followed by byte value

        Args:
            root: Root of Huffman tree

        Returns:
            Serialized tree as bytes
        """
        if not root:
            return b''

        result = bytearray()

        def serialize_node(node: HuffmanNode):
            if node.is_leaf():
                result.append(1)  # Leaf marker
                result.append(node.char)  # Byte value
            else:
                result.append(0)  # Internal node marker
                if node.left:
                    serialize_node(node.left)
                if node.right:
                    serialize_node(node.right)

        serialize_node(root)
        return bytes(result)

    def deserialize_tree(self, data: bytes) -> Tuple[HuffmanNode, int]:
        """
        Deserialize Huffman tree from bytes.

        Args:
            data: Serialized tree data

        Returns:
            Tuple of (root node, bytes consumed)
        """
        index = [0]  # Use list for mutable reference

        def deserialize_node() -> HuffmanNode:
            if index[0] >= len(data):
                return None

            marker = data[index[0]]
            index[0] += 1

            if marker == 1:  # Leaf node
                if index[0] >= len(data):
                    return None
                char = data[index[0]]
                index[0] += 1
                return HuffmanNode(char=char)
            else:  # Internal node
                left = deserialize_node()
                right = deserialize_node()
                return HuffmanNode(left=left, right=right)

        root = deserialize_node()
        return root, index[0]

    def bits_to_bytes(self, bit_string: str) -> bytes:
        """
        Convert bit string to bytes, padding with zeros if necessary.

        Args:
            bit_string: String of '0' and '1' characters

        Returns:
            Bytes representation
        """
        # Pad to make length multiple of 8
        padding = 8 - (len(bit_string) % 8)
        if padding != 8:
            bit_string += '0' * padding

        # Convert to bytes
        result = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_str = bit_string[i:i+8]
            byte_val = int(byte_str, 2)
            result.append(byte_val)

        return bytes(result), padding

    def compress_data(self, data: bytes) -> bytes:
        """
        Compress data using Huffman coding.

        File format:
        - Tree size (4 bytes)
        - Serialized tree
        - Padding bits (1 byte)
        - Original data length (4 bytes)
        - Encoded data

        Args:
            data: Input bytes to compress

        Returns:
            Compressed bytes
        """
        if not data:
            return b''

        print(f"Compressing {len(data)} bytes using Huffman coding...")

        # Build frequency table and tree
        frequencies = self.build_frequency_table(data)
        tree = self.build_huffman_tree(frequencies)

        if not tree:
            return b''

        # Generate codes and encode data
        codes = self.generate_codes(tree)
        encoded_bits = self.encode_data(data, codes)

        # Convert bits to bytes
        encoded_bytes, padding = self.bits_to_bytes(encoded_bits)

        # Serialize tree
        tree_data = self.serialize_tree(tree)

        # Create compressed file
        result = bytearray()

        # Tree size (4 bytes)
        result.extend(len(tree_data).to_bytes(4, 'big'))

        # Serialized tree
        result.extend(tree_data)

        # Padding (1 byte)
        result.append(padding)

        # Original data length (4 bytes)
        result.extend(len(data).to_bytes(4, 'big'))

        # Encoded data
        result.extend(encoded_bytes)

        print(
            f"Huffman compression complete: {len(data)} → {len(result)} bytes")
        return bytes(result)

    @measure_compression_time
    def compress_file(self, input_filepath: str, output_filepath: str) -> int:
        """
        Compress a file using Huffman coding.

        Args:
            input_filepath: Path to input file
            output_filepath: Path to output compressed file

        Returns:
            Size of compressed file
        """
        print(
            f"\n=== Huffman Compression: {os.path.basename(input_filepath)} ===")

        # Read input file
        data = read_file_binary(input_filepath)
        original_size = len(data)

        # Compress data
        compressed_data = self.compress_data(data)
        compressed_size = len(compressed_data)

        # Write compressed file
        write_file_binary(output_filepath, compressed_data)

        return compressed_size

    def analyze_compression_potential(self, data: bytes) -> dict:
        """
        Analyze entropy and compression potential.

        Args:
            data: Input data to analyze

        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}

        frequencies = Counter(data)
        total_bytes = len(data)

        # Calculate entropy
        entropy = 0
        for freq in frequencies.values():
            probability = freq / total_bytes
            entropy -= probability * (probability.bit_length() - 1)

        # Calculate average code length with Huffman coding
        # This is a theoretical minimum based on entropy
        theoretical_compression = entropy / 8  # Convert bits to bytes ratio

        return {
            'unique_bytes': len(frequencies),
            'entropy': entropy,
            'theoretical_compression_ratio': (1 - theoretical_compression) * 100,
            'most_frequent_byte_percentage': max(frequencies.values()) / total_bytes * 100
        }


def compress_all_test_files():
    """Compress all files in the test directory using Huffman coding."""
    compressor = HuffmanCompressor()
    test_dir = "files_to_compress"
    output_dir = "compressed_output/huffman"

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
        output_file = os.path.join(output_dir, f"{filename}.huff")

        try:
            original_size = get_file_size(input_file)
            compressed_size, compression_time = compressor.compress_file(
                input_file, output_file)

            stats = CompressionStats(
                "Huffman", original_size, compressed_size, compression_time)
            results.append(stats)
            print_compression_stats(stats)

            # Show analysis
            data = read_file_binary(input_file)
            analysis = compressor.analyze_compression_potential(data)
            print("Compression Analysis:")
            print(f"  Unique bytes: {analysis['unique_bytes']}/256")
            print(f"  Entropy: {analysis['entropy']:.2f} bits per byte")
            print(
                f"  Theoretical best: {analysis['theoretical_compression_ratio']:.1f}% compression")
            print()

        except Exception as e:
            print(f"Error compressing {filename}: {e}")

    return results


if __name__ == "__main__":
    print("Huffman Coding Compression Demo")
    print("===============================")
    print()
    print("Huffman coding assigns shorter codes to frequent bytes:")
    print("- Builds a binary tree based on byte frequencies")
    print("- Frequent bytes get shorter codes (closer to root)")
    print("- Rare bytes get longer codes (deeper in tree)")
    print()
    print("Best for: Text files, files with uneven byte distribution")
    print("Optimal: Theoretically optimal for given frequency distribution")
    print()

    results = compress_all_test_files()

    if results:
        print("\n" + "="*60)
        print("HUFFMAN COMPRESSION SUMMARY")
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
            f"\nBest compression: {best.compression_ratio:.1f}% on file with uneven distribution")
        print(
            f"Worst compression: {worst.compression_ratio:.1f}% on file with uniform distribution")
