# mojo-levenshtein

`python-Levenshtein`-compatible edit distance and similarity ratio, with the
compute-intensive kernels implemented in [Mojo](https://www.modular.com/mojo).
The Python package keeps the upstream function names, signatures, sequence
semantics, processors, weights, and cutoffs for the covered subset.

```python
import mojo_levenshtein as Levenshtein

assert Levenshtein.distance("lewenstein", "levenshtein") == 2
assert Levenshtein.distance(
    "lewenstein", "levenshtein", weights=(1, 1, 2)
) == 3
assert Levenshtein.ratio("lewenstein", "levenshtein") == 0.8571428571428572
```

The implementation is independent and MIT-licensed. `python-Levenshtein` is a
development dependency used as the behavioral oracle in tests and benchmarks.

## Coverage

| API | Status |
| --- | --- |
| `distance(s1, s2, *, weights=(1, 1, 1), processor=None, score_cutoff=None, score_hint=None)` | Covered |
| `ratio(s1, s2, *, processor=None, score_cutoff=None)` | Covered |
| `str`, `bytes`, and arbitrary sequences of hashable values | Covered |
| Weighted insertion, deletion, and substitution costs | Covered |
| `hamming`, `jaro`, `jaro_winkler` | Not covered |
| `editops`, `opcodes`, `matching_blocks`, `apply_edit` | Not covered |
| `median`, `median_improve`, `quickmedian`, `seqratio`, `setratio` | Not covered |

The package accepts the same broad sequence model as upstream, including
Unicode code points, byte values, NumPy arrays, and lists or tuples containing
hashable tokens. Unlike upstream's unsigned result domain, the Mojo ABI and
weighted-DP row use signed 64-bit integers. Weights above `2**63 - 1`, and
weighted comparisons whose safe dynamic-programming bound exceeds that value,
raise `OverflowError` instead of narrowing or wrapping.

## Install and run

This repository currently supports Linux source checkouts; it is not a
prebuilt PyPI wheel. Install the pinned Mojo and Python environment with Pixi,
then build the shared library:

```bash
pixi install
pixi run build
pixi run python -c \
  'import mojo_levenshtein as L; print(L.distance("kitten", "sitting"), L.ratio("kitten", "sitting"))'
```

This prints:

```text
3 0.6153846153846154
```

The normal development commands are:

```bash
pixi run build
pixi run test
pixi run bench
```

The Python loader also builds the library on first import when the source is
newer than `dist/libmojo-levenshtein.so`. A deployed package can set
`MOJO_LEVENSHTEIN_LIB` to an already-built shared library.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, using
Python 3.13.14 and python-Levenshtein 0.27.3. Times include Python-side sequence
encoding, scratch allocation, and the FFI call. The benchmark reports the best
of five runs on identical inputs.

| operation | Mojo | python-Levenshtein | upstream / Mojo |
| --- | ---: | ---: | ---: |
| distance, bytes length 10,000 | 5.940 ms | 5.434 ms | 0.91x |
| distance, Unicode length 4,000 | 1.546 ms | 1.009 ms | 0.65x |
| ratio, bytes length 20,000 | 11.025 ms | 11.958 ms | 1.08x |
| weighted distance, length 2,000 | 9.682 ms | 12.138 ms | 1.25x |

In this run the ratio and weighted cases were faster than upstream, while both
ordinary distance cases were slower. Results are workload- and
machine-specific. No GPU or multithreaded path is included.

## How it works

The common unit-cost distance path uses the multiword Myers bit-vector
algorithm. It processes 64 pattern positions per machine word and supports
patterns longer than one word. Its carry-preserving update is fused into one
pass over two state rows, avoiding three temporary rows. Native-width SIMD
scans common prefixes and suffixes and fills scratch buffers, with scalar tail
loops for every remainder. `ratio` is upstream's normalized indel similarity,
so its kernel computes longest common subsequence length with a multiword
bitset recurrence. Arbitrary nonnegative operation weights use a one-row
dynamic program.

Python converts strings to contiguous UTF-32 code points, bytes to unsigned
values, and general hashable sequences to shared integer symbol IDs. NumPy owns
all input and scratch storage: bit masks are row-major `uint64`, weighted-DP
rows are contiguous `int64`, and the Mojo kernels allocate nothing. Before each
call, the Python boundary checks every buffer's dtype, rank, contiguous layout,
minimum length, and non-null address. Local references keep all NumPy owners
alive for the complete synchronous call.

The shared library exposes non-parametric `@export` functions with
`abi("C")`. Buffers cross `ctypes` as 64-bit integer addresses and are rebuilt
inside Mojo as pointers with `AnyOrigin[mut=True]`. One FFI call performs the
whole comparison.

## Correctness

The pytest suite checks all covered behavior against the real
`python-Levenshtein` package, including randomized inputs, Unicode, bytes,
general hashable sequences, asymmetric and zero weights, processors, score
cutoffs, and bit-vector boundaries at 63/64/65 and 127/128/129 elements.

```bash
pixi run build && pixi run test
```

## License

MIT
