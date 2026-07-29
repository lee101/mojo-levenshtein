"""Load the Mojo shared library and describe its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_LEVENSHTEIN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-levenshtein.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "ml_distance_unit_u32": ([I] * 9, I),
    "ml_distance_weighted_u32": ([I] * 8, I),
    "ml_indel_distance_u32": ([I] * 9, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_LEVENSHTEIN_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_LEVENSHTEIN_LIB does not exist: {LIB}")
    source = os.path.join(SRC, "levenshtein.mojo")
    if not os.path.exists(source):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo source at {source} and no shared library at {LIB}")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(source):
            return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded
