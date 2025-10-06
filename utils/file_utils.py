"""
Utility functions for file operations and compression analysis.
"""

import os
import time
from typing import Tuple, List


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio as percentage reduction."""
    if original_size == 0:
        return 0.0
    return ((original_size - compressed_size) / original_size) * 100


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def read_file_binary(filepath: str) -> bytes:
    """Read file in binary mode."""
    with open(filepath, 'rb') as f:
        return f.read()


def write_file_binary(filepath: str, data: bytes) -> None:
    """Write data to file in binary mode."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(data)


def get_all_test_files(directory: str) -> List[str]:
    """Get all files in the test directory."""
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return sorted(files)


def measure_compression_time(func):
    """Decorator to measure compression time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        compression_time = end_time - start_time
        return result, compression_time
    return wrapper


class CompressionStats:
    """Class to track compression statistics."""

    def __init__(self, method_name: str, original_size: int, compressed_size: int,
                 compression_time: float = 0.0):
        self.method_name = method_name
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.compression_time = compression_time
        self.compression_ratio = calculate_compression_ratio(
            original_size, compressed_size)

    def __str__(self):
        return (f"{self.method_name}: "
                f"{format_size(self.original_size)} → {format_size(self.compressed_size)} "
                f"({self.compression_ratio:.1f}% reduction) "
                f"in {self.compression_time:.2f}s")


def print_compression_stats(stats: CompressionStats):
    """Print formatted compression statistics."""
    print(f"Method: {stats.method_name}")
    print(f"Original size: {format_size(stats.original_size)}")
    print(f"Compressed size: {format_size(stats.compressed_size)}")
    print(f"Compression ratio: {stats.compression_ratio:.2f}%")
    print(f"Time taken: {stats.compression_time:.2f} seconds")
    print("-" * 50)
