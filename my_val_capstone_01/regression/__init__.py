"""Household contents-value regression.

The notebooks are in notebooks/regression (01_ky to 07_ky).
This package is separate from my_val_capstone_01, the web front end and next-category model.
"""

import ctypes
from pathlib import Path


def _preload_openmp() -> None:
    """LightGBM's wheel links the system OpenMP runtime (libgomp.so.1), which the Vercel function runtime
    does not ship, so loading the saved LightGBM model fails there. Use the system copy when present,
    otherwise the bundled copy, so the loader finds it already loaded."""
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        bundled = Path(__file__).resolve().parent / "_native" / "libgomp.so.1"
        if bundled.exists():
            ctypes.CDLL(str(bundled), mode=ctypes.RTLD_GLOBAL)


_preload_openmp()
