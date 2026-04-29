"""Local-build setup for the ncolor C++/threadpool prototype.

Build in place::

    cd <ncolor>/cpp_proto
    pip install -e . --config-settings editable_mode=compat

Or build the extension only without installing::

    python setup.py build_ext --inplace

A ``build_ext`` post-hook runs the SMT/HT calibration and writes
``~/.cache/ncolor_cpp_proto/smt_threads.json`` so that ``Solver`` users
can call ``_smt.auto_threads()`` with a sub-millisecond cache lookup.
"""
from __future__ import annotations

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    import pybind11
except ImportError as exc:
    sys.exit(
        "pybind11 is required to build ncolor_cpp_proto. "
        "Install with: pip install pybind11"
    )

extra_compile_args = []
extra_link_args = []
if sys.platform == "win32":
    # MSVC: /std:c++17, /O2 max optimization, /arch:AVX2 to enable AVX2
    # codegen for the parabolic kernel's FMA inner loop (matches the GCC
    # `-march=native` win on Zen/Skylake hosts).
    extra_compile_args += ["/std:c++17", "/O2", "/arch:AVX2", "/EHsc"]
else:
    extra_compile_args += [
        "-std=c++17",
        "-O3",
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
    "ncolor_cpp_proto._ncolor_cpp_proto_impl",
    sources=["binding.cpp"],
    include_dirs=[pybind11.get_include(), "."],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

class build_ext(_build_ext):
    """build_ext + post-build SMT/HT thread-count calibration.

    After the .so is built, import ``_smt`` from the build output directory
    and run a calibration on a 1024² mask (~300-450 ms). The result is
    written to ``~/.cache/ncolor_cpp_proto/smt_threads.json`` keyed by
    hostname + CPU model. Subsequent ``_smt.auto_threads()`` calls hit the
    cache in <1 ms.

    Always re-runs (even if cache exists) so install/rebuild gets fresh
    timings — useful if CPU/firmware/thermal-policy changed since last
    install. Skips silently if numpy isn't yet available (pip build
    isolation). Set the env var ``NCOLOR_NO_CALIBRATE=1`` to disable
    entirely (useful for CI / cross-compilation).
    """

    def run(self):
        super().run()
        if os.environ.get("NCOLOR_NO_CALIBRATE"):
            return
        srcdir = os.path.dirname(os.path.abspath(__file__))
        # Candidate roots that contain the ``ncolor_cpp_proto`` package after
        # build. For ``--inplace`` it's ``src/``; for non-inplace it's
        # ``build_lib``. The package's ``__init__.py`` handles network-mount
        # ``.so`` loading transparently (smbfs/UNC), so calibration imports
        # never touch the NAS-resident binary directly.
        candidates = [p for p in (os.path.join(srcdir, "src"), self.build_lib)
                      if p and p not in sys.path]
        for p in candidates: sys.path.insert(0, p)
        try:
            try:
                import numpy  # noqa: F401  — needed by calibrate()
            except ImportError:
                print("[ncolor_cpp_proto] numpy not available at build time; "
                      "SMT calibration deferred to first auto_threads() call.")
                return
            try:
                from ncolor_cpp_proto import _smt
                _smt.calibrate(force=True, verbose=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ncolor_cpp_proto] SMT calibration skipped ({exc!r}); "
                      "auto_threads() will fall back to physical core count.")
        finally:
            for p in candidates:
                try: sys.path.remove(p)
                except ValueError: pass


setup(
    name="ncolor_cpp_proto",
    version="0.1.0",
    description="C++/threadpool prototype of ncolor's _search_hashset_parallel.",
    ext_modules=[ext],
    package_dir={"": "src"},
    packages=["ncolor_cpp_proto"],
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.10", "numpy>=1.20", "platformdirs"],
    python_requires=">=3.10",
)
