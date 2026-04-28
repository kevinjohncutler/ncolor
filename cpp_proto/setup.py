"""Local-build setup for the ncolor C++/threadpool prototype.

Build in place::

    cd <ncolor>/cpp_proto
    pip install -e . --config-settings editable_mode=compat

Or build the extension only without installing::

    python setup.py build_ext --inplace
"""
from __future__ import annotations

import sys
from setuptools import setup, Extension

try:
    import pybind11
except ImportError as exc:
    sys.exit(
        "pybind11 is required to build ncolor_cpp_proto. "
        "Install with: pip install pybind11"
    )

extra_compile_args = ["-std=c++17", "-O3"]
extra_link_args = []
if sys.platform != "win32":
    extra_compile_args += [
        "-fPIC",
        "-pthread",
        # Let the compiler pick the best ISA for the host: AVX2/AVX512 on Zen
        # workstations, NEON on Apple Silicon. The serial parabolic-envelope
        # build loop is data-dependent so it won't auto-vectorize across
        # iterations, but `-march=native` improves codegen for the FP arithmetic
        # (single-pass FMAs, fewer mov/cvt) and makes the second-phase scan
        # loop vectorize cleanly.
        "-march=native",
        "-ffp-contract=fast",  # fuse mul+add → FMA (matches LLVM-on-numba's FMA emission)
        "-funroll-loops",
    ]
    extra_link_args += ["-pthread"]
    if sys.platform == "darwin":
        extra_compile_args += ["-mmacosx-version-min=10.14"]

ext = Extension(
    "ncolor_cpp_proto",
    sources=["binding.cpp"],
    include_dirs=[pybind11.get_include(), "."],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="ncolor_cpp_proto",
    version="0.1.0",
    description="C++/threadpool prototype of ncolor's _search_hashset_parallel.",
    ext_modules=[ext],
    py_modules=[],
    install_requires=["pybind11>=2.10", "numpy>=1.20"],
    python_requires=">=3.10",
)
