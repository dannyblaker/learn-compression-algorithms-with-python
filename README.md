# Learn Compression Algorithms with Python

This repository demonstrates various prominent compression algorithms implemented in Python. Each method includes a detailed implementation that shows the compression process step-by-step. I have avoided using external compression libraries to maximize educational value.

# Background

I originally wrote this repository to test my own ability to describe, articulate and dissect various popular compression algorithms and their components while practicing for coding job interviews. I decided it may be useful to others, thus I cleaned, polished and published it to Github.

**You are most welcome to use this code in your commercial projects, all that I ask in return is that you credit my work by providing a link back to this repository. Thank you & Enjoy!**

## Repository Structure

```
├── README.md
├── files_to_compress/          # Test files for compression
├── compression_methods/        # Individual compression implementations
│   ├── huffman_compression.py
│   ├── lz77_compression.py
│   ├── lzw_compression.py
│   ├── rle_compression.py
│   ├── arithmetic_compression.py
│   └── burrows_wheeler.py
├── compressed_output/          # Output directory for compressed files
├── utils/                      # Utility functions
│   └── file_utils.py
├── run_all_compression.py      # Script to run all methods and generate report
└── compression_report.csv      # Generated comparison report
```

## Compression Methods Implemented

### 1. Run-Length Encoding (RLE)
**File**: `compression_methods/rle_compression.py`
- Simple compression technique that replaces sequences of identical bytes
- Best for files with many repeated characters
- Educational baseline for understanding compression concepts

### 2. Huffman Coding
**File**: `compression_methods/huffman_compression.py`
- Variable-length prefix coding based on character frequency
- Builds a binary tree to assign shorter codes to frequent characters
- Optimal for files with uneven character distribution

### 3. LZ77 Compression
**File**: `compression_methods/lz77_compression.py`
- Dictionary-based compression using sliding window
- Replaces repeated substrings with references to previous occurrences
- Foundation for many modern compression algorithms (like gzip)

### 4. LZW Compression
**File**: `compression_methods/lzw_compression.py`
- Dictionary-based compression that builds a dictionary dynamically
- Used in GIF files and Unix compress utility
- Adaptive algorithm that learns patterns as it processes data

### 5. Arithmetic Coding
**File**: `compression_methods/arithmetic_compression.py`
- Entropy encoding that represents entire messages as single numbers
- Theoretically optimal compression for given probability model
- More complex but can achieve better compression than Huffman

### 6. Burrows-Wheeler Transform + Move-to-Front
**File**: `compression_methods/burrows_wheeler.py`
- Transform that rearranges data to make it more compressible
- Combined with Move-to-Front encoding and RLE
- Used in bzip2 compression algorithm

## Usage

### Setup Test files

- create the folder `files_to_compress` in the project's root folder
- drop in the files you wish to compress (I recommend very small files so tests run faster)

### Running Individual Compression Methods

Each compression method can be run independently:

```bash
python compression_methods/huffman_compression.py
python compression_methods/lz77_compression.py
python compression_methods/lzw_compression.py
python compression_methods/rle_compression.py
python compression_methods/arithmetic_compression.py
python compression_methods/burrows_wheeler.py
```

### Running All Methods and Generating Report

To run all compression methods on all test files and generate a CSV report:

```bash
python run_all_compression.py
```

This will:
1. Apply each compression method to every file in `files_to_compress/`
2. Save compressed files to `compressed_output/`
3. Generate `compression_report.csv` with detailed size comparisons

## Report Format

The generated CSV report includes:
- Original filename
- Original file size
- Compressed size for each method
- Compression ratio for each method
- Best performing method for each file

## Educational Notes

Each compression script includes:
- Detailed comments explaining the algorithm
- Step-by-step process visualization
- Performance metrics and analysis
- Strengths and weaknesses of each approach

## Requirements

- Python 3.7+
- No external compression libraries required
- Standard library modules: `os`, `sys`, `csv`, `heapq`, `collections`, `math`

## Implementation Philosophy

This repository prioritizes educational value over performance:
- Pure Python implementations for transparency
- Detailed logging of compression steps
- Clear, readable code with extensive comments
- Focus on understanding algorithm mechanics rather than optimization

## Performance Expectations

These implementations are educational and not optimized for production use. Real-world compression tools use:
- Optimized C/C++ implementations
- Hardware acceleration
- Advanced algorithmic improvements
- Streaming processing for large files

For production compression needs, use established libraries like `gzip`, `bzip2`, or `lzma`.