"""Benchmark Mojo kernels against python-Levenshtein on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import Levenshtein as upstream  # noqa: E402
import mojo_levenshtein as mojo  # noqa: E402


def best_time(function, repetitions=5):
    best = math.inf
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mutate(data: bytes, every: int) -> bytes:
    changed = bytearray(data)
    for index in range(every // 2, len(changed), every):
        changed[index] = (changed[index] + 7) & 255
    return bytes(changed)


def main() -> None:
    rng = np.random.default_rng(2026)
    base_10k = rng.integers(97, 123, size=10_000, dtype=np.uint8).tobytes()
    changed_10k = mutate(base_10k, 97)
    base_20k = rng.integers(97, 123, size=20_000, dtype=np.uint8).tobytes()
    changed_20k = mutate(base_20k, 71)
    unicode_a = ("Mojo λ distance 東京 café " * 180)[:4_000]
    unicode_b = unicode_a.replace("λ", "δ").replace("é", "e")
    weighted_a = rng.integers(97, 103, size=2_000, dtype=np.uint8).tobytes()
    weighted_b = mutate(weighted_a, 19)

    cases = [
        (
            "distance, bytes length 10,000",
            lambda: mojo.distance(base_10k, changed_10k),
            lambda: upstream.distance(base_10k, changed_10k),
        ),
        (
            "distance, Unicode length 4,000",
            lambda: mojo.distance(unicode_a, unicode_b),
            lambda: upstream.distance(unicode_a, unicode_b),
        ),
        (
            "ratio, bytes length 20,000",
            lambda: mojo.ratio(base_20k, changed_20k),
            lambda: upstream.ratio(base_20k, changed_20k),
        ),
        (
            "weighted distance, length 2,000",
            lambda: mojo.distance(weighted_a, weighted_b, weights=(1, 2, 3)),
            lambda: upstream.distance(weighted_a, weighted_b, weights=(1, 2, 3)),
        ),
    ]

    print(f"Machine: {cpu_name()}")
    print(f"Python: {platform.python_version()}; python-Levenshtein: {upstream.__version__}")
    print()
    print("| operation | Mojo | python-Levenshtein | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_seconds, mojo_result = best_time(ours)
        upstream_seconds, upstream_result = best_time(theirs)
        if isinstance(mojo_result, float):
            assert math.isclose(mojo_result, upstream_result, rel_tol=0, abs_tol=1e-15)
        else:
            assert mojo_result == upstream_result
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.3f} ms | "
            f"{upstream_seconds * 1e3:.3f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
