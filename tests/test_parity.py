from __future__ import annotations

import inspect
import random

import numpy as np
import pytest

import Levenshtein as upstream
import mojo_levenshtein as mojo
from mojo_levenshtein import _address


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("lewenstein", "levenshtein", 2),
        ("kitten", "sitting", 3),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
    ],
)
def test_published_and_standard_distance_vectors(left, right, expected):
    assert mojo.distance(left, right) == expected
    assert mojo.distance(left, right) == upstream.distance(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("café", "coffee"),
        ("𐐷mojo", "𐐷Mojo"),
        ("naïve", "naive"),
        ("東京", "京都"),
        ("\0alpha", "\0alpHa"),
        ("\ud800alpha", "\ud800alpHa"),
    ],
)
def test_unicode_parity(left, right):
    assert mojo.distance(left, right) == upstream.distance(left, right)
    assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"abc\0def", b"abc\0dEf"),
        (bytes(range(256)), bytes(reversed(range(256)))),
        ("é", "é".encode()),
        (b"a", ["a"]),
    ],
)
def test_bytes_and_mixed_sequence_parity(left, right):
    assert mojo.distance(left, right) == upstream.distance(left, right)
    assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))


def test_hashable_sequence_parity():
    pairs = [
        ([1, 2, 1, True], (1, 3, 1, 1)),
        (("alpha", "beta", "alpha"), ("alpha", "gamma")),
        (np.array([4, 2, 4, 8]), np.array([4, 4, 8])),
        ("ab", [97, 98]),
        ("ab", ["a", "b"]),
    ]
    for left, right in pairs:
        assert mojo.distance(left, right) == upstream.distance(left, right)
        assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))


def test_randomized_distance_and_ratio_parity():
    rng = random.Random(8743)
    for _ in range(250):
        left = "".join(rng.choice("abcde") for _ in range(rng.randrange(80)))
        right = "".join(rng.choice("abcde") for _ in range(rng.randrange(80)))
        assert mojo.distance(left, right) == upstream.distance(left, right)
        assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))


@pytest.mark.parametrize("length", [63, 64, 65, 127, 128, 129, 257])
def test_bit_vector_word_boundaries(length):
    left = ("abcdefgh" * ((length + 7) // 8))[:length]
    right = list(left)
    for index in range(3, length, 17):
        right[index] = "z"
    right = "".join(right) + "tail"
    assert mojo.distance(left, right) == upstream.distance(left, right)
    assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))


@pytest.mark.parametrize(
    "common_length", [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17]
)
def test_simd_trim_prefix_suffix_tails(common_length):
    prefix = "a" * common_length
    suffix = "b" * common_length
    left = prefix + "x" + suffix
    right = prefix + "yz" + suffix
    assert mojo.distance(left, right) == upstream.distance(left, right)
    assert mojo.ratio(left, right) == pytest.approx(upstream.ratio(left, right))
    assert mojo.distance(left, right, weights=(2, 3, 4)) == upstream.distance(
        left, right, weights=(2, 3, 4)
    )


@pytest.mark.parametrize("match_run", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
def test_wavefront_simd_match_extension_tails(match_run):
    repeated = "abcdefghijklmnopq"[:match_run]
    repetitions = 600 // (match_run + 1) + 1
    left = (repeated + "x") * repetitions + repeated
    right = (repeated + "y") * repetitions + repeated
    assert mojo.distance(left, right) == upstream.distance(left, right)


def test_wavefront_limit_falls_back_to_multiword_myers():
    left = (b"a" * 513) + bytes(range(256))
    right = (b"b" * 513) + bytes(reversed(range(256)))
    assert upstream.distance(left, right) > 384
    assert mojo.distance(left, right) == upstream.distance(left, right)


@pytest.mark.parametrize(
    "weights",
    [
        (1, 1, 2),
        (2, 3, 4),
        (0, 1, 2),
        (1, 0, 2),
        (0, 0, 0),
        (7, 2, 9),
    ],
)
def test_weighted_distance_parity(weights):
    pairs = [
        ("lewenstein", "levenshtein"),
        ("", "inserted"),
        ("deleted", ""),
        ("saturday", "sunday"),
        ("same-prefix-one", "same-prefix-two"),
    ]
    for left, right in pairs:
        assert mojo.distance(left, right, weights=weights) == upstream.distance(
            left, right, weights=weights
        )


def test_randomized_weighted_parity():
    rng = random.Random(901)
    for _ in range(150):
        left = "".join(rng.choice("abcd") for _ in range(rng.randrange(18)))
        right = "".join(rng.choice("abcd") for _ in range(rng.randrange(18)))
        weights = tuple(rng.randrange(5) for _ in range(3))
        assert mojo.distance(left, right, weights=weights) == upstream.distance(
            left, right, weights=weights
        )


def test_asymmetric_insert_delete_costs():
    weights = (2, 7, 4)
    assert mojo.distance("", "abc", weights=weights) == 6
    assert mojo.distance("abc", "", weights=weights) == 21
    assert mojo.distance("abc", "a", weights=weights) == upstream.distance(
        "abc", "a", weights=weights
    )


@pytest.mark.parametrize("cutoff", [0, 1, 2, 5, 100, 2.9])
def test_distance_cutoff_parity(cutoff):
    for left, right in [("kitten", "sitting"), ("abc", "abc"), ("", "abcd")]:
        assert mojo.distance(left, right, score_cutoff=cutoff) == upstream.distance(
            left, right, score_cutoff=cutoff
        )


@pytest.mark.parametrize("cutoff", [0.0, 0.4, 0.8, 0.9, 1.0])
def test_ratio_cutoff_parity(cutoff):
    for left, right in [("lewenstein", "levenshtein"), ("abc", "abc"), ("", "")]:
        assert mojo.ratio(left, right, score_cutoff=cutoff) == pytest.approx(
            upstream.ratio(left, right, score_cutoff=cutoff)
        )


def test_processor_parity_and_call_contract():
    calls: list[object] = []

    def processor(value):
        calls.append(value)
        return value.strip().lower()

    assert mojo.distance(" Alpha ", "ALPINE", processor=processor) == upstream.distance(
        " Alpha ", "ALPINE", processor=lambda value: value.strip().lower()
    )
    assert calls == [" Alpha ", "ALPINE"]
    assert mojo.ratio(["lewenstein"], ["levenshtein"], processor=lambda x: x[0]) == (
        pytest.approx(
            upstream.ratio(
                ["lewenstein"], ["levenshtein"], processor=lambda x: x[0]
            )
        )
    )


def test_score_hint_does_not_change_distance():
    expected = upstream.distance("a" * 200 + "x", "a" * 200 + "y", score_hint=1)
    assert mojo.distance("a" * 200 + "x", "a" * 200 + "y", score_hint=1) == expected


@pytest.mark.parametrize(
    "weights",
    [(1, 2), (1, 2, 3, 4), (1, -1, 1), ("1", 1, 1)],
)
def test_invalid_weights_raise_like_upstream(weights):
    with pytest.raises(type(_captured_error(upstream.distance, weights=weights))):
        mojo.distance("a", "b", weights=weights)


def _captured_error(fn, **kwargs):
    try:
        fn("a", "b", **kwargs)
    except Exception as error:
        return error
    raise AssertionError("upstream unexpectedly accepted invalid input")


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, "bad"])
def test_invalid_ratio_cutoff_raises_type_error(cutoff):
    with pytest.raises(TypeError):
        mojo.ratio("a", "b", score_cutoff=cutoff)


def test_unhashable_elements_raise_type_error():
    with pytest.raises(TypeError):
        mojo.distance([[1]], [[2]])
    with pytest.raises(TypeError):
        mojo.ratio([[1]], [[2]])


def test_signatures_match_covered_upstream_api():
    assert str(inspect.signature(mojo.distance)) == str(
        inspect.signature(upstream.distance)
    )
    assert str(inspect.signature(mojo.ratio)) == str(inspect.signature(upstream.ratio))


def test_weighted_abi_overflow_is_rejected_instead_of_wrapping():
    with pytest.raises(OverflowError, match="signed 64-bit Mojo ABI"):
        mojo.distance("aa", "", weights=(1, np.iinfo(np.int64).max, 1))
    with pytest.raises(OverflowError, match="signed 64-bit Mojo ABI"):
        mojo.distance("aa", "b", weights=(np.iinfo(np.int64).max, 1, 1))


@pytest.mark.parametrize(
    "array",
    [
        np.zeros(3, dtype=np.uint64),
        np.zeros((1, 3), dtype=np.uint32),
        np.zeros(6, dtype=np.uint32)[::2],
        np.zeros(2, dtype=np.uint32),
    ],
)
def test_ffi_buffer_contract_rejects_wrong_dtype_shape_stride_or_length(array):
    with pytest.raises(RuntimeError, match="invalid FFI buffer"):
        _address(array, np.uint32, 3)
