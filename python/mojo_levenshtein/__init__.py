"""Python-compatible edit distance and ratio backed by Mojo kernels."""

from __future__ import annotations

from numbers import Real
from typing import Any, Hashable, Sequence

import numpy as np

from ._lib import lib


__all__ = ["distance", "ratio"]
__version__ = "0.1.0"

_Weights = tuple[int, int, int]
_ABI_INT_MAX = np.iinfo(np.int64).max
_U32_SYMBOL_LIMIT = np.iinfo(np.uint32).max
_WAVEFRONT_STORAGE = 2 * (2 * 384 + 3)


def _canonical_token(value: Hashable) -> Hashable:
    if isinstance(value, str) and len(value) == 1:
        return ord(value)
    return value


def _mapped_pair(
    s1: Sequence[Hashable], s2: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    mapping: dict[Hashable, int] = {}
    first: list[int] = []
    second: list[int] = []
    for source, target in ((s1, first), (s2, second)):
        for value in source:
            token = _canonical_token(value)
            try:
                target.append(mapping[token])
            except KeyError:
                code = len(mapping)
                if code > _U32_SYMBOL_LIMIT:
                    raise OverflowError("too many distinct sequence elements for uint32 ABI")
                mapping[token] = code
                target.append(code)
    a = np.asarray(first, dtype=np.uint32)
    b = np.asarray(second, dtype=np.uint32)
    return a, b, len(set(first)), len(set(second))


def _encode_pair(
    s1: Sequence[Hashable], s2: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    len(s1)
    len(s2)
    simple = isinstance(s1, (str, bytes)) and isinstance(s2, (str, bytes))
    if not simple:
        return _mapped_pair(s1, s2)

    def encode(value: str | bytes) -> tuple[np.ndarray, int]:
        if isinstance(value, str):
            raw = value.encode("utf-32-le", errors="surrogatepass")
            return np.frombuffer(raw, dtype="<u4"), len(set(value))
        raw8 = np.frombuffer(value, dtype=np.uint8)
        return raw8.astype(np.uint32), len(set(value))

    a, unique_a = encode(s1)
    b, unique_b = encode(s2)
    return a, b, unique_a, unique_b


def _address(
    array: np.ndarray, dtype: np.dtype, minimum_length: int
) -> int:
    """Validate an ABI buffer and return its address while its owner is live."""
    expected = np.dtype(dtype)
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 1
        or array.dtype != expected
        or not array.flags.c_contiguous
        or array.size < minimum_length
    ):
        raise RuntimeError(
            f"invalid FFI buffer: expected contiguous 1-D {expected} "
            f"with at least {minimum_length} elements"
        )
    address = int(array.ctypes.data)
    if address == 0:
        raise RuntimeError("invalid FFI buffer: null address")
    return address


def _bit_scratch(
    unique: int, length: int, work_factor: int, minimum_work: int = 1
):
    words = (length + 63) // 64
    hash_size = 1
    while hash_size < max(2, unique * 2):
        hash_size <<= 1
    masks = np.zeros(max(1, (unique + 1) * words), dtype=np.uint64)
    keys = np.empty(hash_size, dtype=np.uint32)
    values = np.empty(hash_size, dtype=np.int64)
    work = np.empty(max(minimum_work, work_factor * words), dtype=np.uint64)
    return masks, keys, values, hash_size, work


def _coerce_weight(value: Any) -> int:
    if not isinstance(value, Real):
        raise TypeError("an integer is required")
    result = int(value)
    if result < 0:
        raise OverflowError("can't convert negative value to size_t")
    if result > _ABI_INT_MAX:
        raise OverflowError("weight exceeds the signed 64-bit Mojo ABI")
    return result


def _weights(value: Any) -> _Weights:
    if value is None:
        return 1, 1, 1
    insertion, deletion, substitution = value
    return (
        _coerce_weight(insertion),
        _coerce_weight(deletion),
        _coerce_weight(substitution),
    )


def _distance_cutoff(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise TypeError("an integer is required")
    result = int(value)
    if result < 0:
        raise OverflowError("can't convert negative value to uint64_t")
    return result


def distance(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    """Calculate weighted Levenshtein edit distance.

    The public signature and cutoff behavior match
    ``Levenshtein.distance`` for the covered sequence inputs.
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)
    insertion, deletion, substitution = _weights(weights)
    cutoff = _distance_cutoff(score_cutoff)
    if score_hint is not None:
        _distance_cutoff(score_hint)
    a, b, unique_a, unique_b = _encode_pair(s1, s2)
    n, m = len(a), len(b)

    if (insertion, deletion, substitution) == (1, 1, 1):
        if m > n:
            a, b = b, a
            n, m = m, n
            unique_b = unique_a
        if m == 0:
            result = n
        else:
            use_wavefront = min(n, m) >= 512
            masks, keys, values, hash_size, work = _bit_scratch(
                unique_b,
                m,
                2,
                _WAVEFRONT_STORAGE if use_wavefront else 1,
            )
            result = int(
                lib().ml_distance_unit_u32(
                    _address(a, np.uint32, n),
                    n,
                    _address(b, np.uint32, m),
                    m,
                    _address(masks, np.uint64, (unique_b + 1) * ((m + 63) // 64)),
                    _address(keys, np.uint32, hash_size),
                    _address(values, np.int64, hash_size),
                    hash_size,
                    _address(work, np.uint64, 2 * ((m + 63) // 64)),
                    use_wavefront,
                )
            )
    else:
        if m > n:
            a, b = b, a
            n, m = m, n
            insertion, deletion = deletion, insertion
        if m == 0:
            if n and deletion > _ABI_INT_MAX // n:
                raise OverflowError("distance exceeds the signed 64-bit Mojo ABI")
            result = n * deletion
        else:
            # Every DP cell is bounded by deleting its prefix and inserting the
            # other one. Keeping that bound representable prevents signed
            # overflow in both row initialization and recurrence additions.
            if (
                (n and deletion > _ABI_INT_MAX // n)
                or (m and insertion > _ABI_INT_MAX // m)
                or n * deletion > _ABI_INT_MAX - m * insertion
            ):
                raise OverflowError("distance computation exceeds the signed 64-bit Mojo ABI")
            row = np.empty(m + 1, dtype=np.int64)
            result = int(
                lib().ml_distance_weighted_u32(
                    _address(a, np.uint32, n),
                    n,
                    _address(b, np.uint32, m),
                    m,
                    insertion,
                    deletion,
                    substitution,
                    _address(row, np.int64, m + 1),
                )
            )
    if cutoff is not None and result > cutoff:
        return cutoff + 1
    return result


def ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """Calculate normalized insertion/deletion similarity in ``[0, 1]``."""
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)
    cutoff = 0.0 if score_cutoff is None else score_cutoff
    if not isinstance(cutoff, Real):
        raise TypeError(f"must be real number, not {type(cutoff).__name__}")
    cutoff = float(cutoff)
    if cutoff < 0.0 or cutoff > 1.0:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 1.0")

    a, b, unique_a, unique_b = _encode_pair(s1, s2)
    n, m = len(a), len(b)
    total = n + m
    if total == 0:
        result = 1.0
    elif n == 0 or m == 0:
        result = 0.0
    else:
        if m > n:
            a, b = b, a
            n, m = m, n
            unique_b = unique_a
        masks, keys, values, hash_size, state = _bit_scratch(unique_b, m, 1)
        indel = int(
            lib().ml_indel_distance_u32(
                _address(a, np.uint32, n),
                n,
                _address(b, np.uint32, m),
                m,
                _address(masks, np.uint64, (unique_b + 1) * ((m + 63) // 64)),
                _address(keys, np.uint32, hash_size),
                _address(values, np.int64, hash_size),
                hash_size,
                _address(state, np.uint64, (m + 63) // 64),
            )
        )
        result = 1.0 - indel / total
    return result if result >= cutoff else 0.0
