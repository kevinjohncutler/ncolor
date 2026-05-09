"""ncolor — 4-color label graph coloring + label expansion.

Build pipeline:

  * Pure-Python sources live in ``src/ncolor/``.
  * The C++ engine source lives in ``cpp/``; a single pybind11 extension
    ``ncolor._backend._impl`` is built from ``cpp/binding.cpp`` (which
    pulls in the rest of the headers).
  * After the extension builds, an SMT calibration post-hook times the
    expand kernel at T=physical and T=logical and writes the optimal
    thread count to ``platformdirs.user_cache_dir("ncolor")``. Disable
    with ``NCOLOR_NO_CALIBRATE=1`` (cross-compile / CI).

Windows builds: pass ``NCOLOR_USE_CLANG_CL=1`` to swap distutils' default
``cl.exe`` for ``clang-cl.exe`` — clang-cl's LLVM autovectoriser handles
the L1 inner loops MSVC's auto-vectoriser punts on.
"""
from __future__ import annotations

import os
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    import pybind11
except ImportError:
    sys.exit(
        "pybind11 is required to build ncolor. Install with: pip install pybind11"
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Hard runtime deps for the cpp pipeline. ncolor.label, ncolor.expand_labels,
# ncolor.format_labels, ncolor.connected_components, ncolor.regionprops all
# work with just numpy + platformdirs.
install_deps = [
    "numpy",
    "platformdirs",  # SMT calibration cache + native loader
]

# Optional features:
#   [clean]:  ncolor.format_labels(clean=True) + ncolor.delete_spurs use
#             scikit-image (CCL + remove_small_holes), scipy.ndimage
#             (convolve), and fastremap.renumber/refit. ~140 MB extra disk.
#             Cpp ncolor.connected_components / ncolor.regionprops are
#             available without this extra.
#
# The numba reference implementation was retired in v1.6.0 — the cpp
# engine is the only backend. The numba sources still live in the repo
# at ``legacy_numba/`` (gitignored) for parity-testing if needed.
extras_deps = {
    "clean": ["scipy", "scikit-image", "fastremap"],
}

extra_compile_args = []
extra_link_args = []
USE_CLANG_CL = os.environ.get("NCOLOR_USE_CLANG_CL") == "1"
# -march=native picks AVX2/AVX512 on Zen, NEON 4×4 transpose on Apple
# Silicon, etc. — but produces wheels that crash on older CPUs. Default
# ON for source builds (developer machines), OFF in cibuildwheel where
# we want a portable baseline. The arm64 SIMD paths in expand.hpp are
# gated on ``__aarch64__`` (always true on apple-arm) so they stay on
# regardless of the march flag.
MARCH_NATIVE = os.environ.get("NCOLOR_MARCH_NATIVE", "1") == "1"

if sys.platform == "win32":
    extra_compile_args += ["/std:c++17", "/O2", "/EHsc"]
    if MARCH_NATIVE:
        extra_compile_args += ["/arch:AVX2"]
    if USE_CLANG_CL:
        # clang-cl maps /O2 -> -O2; push to -O3 + (optionally) -march=native
        # via /clang:. Without LTO the host MS link.exe is fine.
        extra_compile_args += [
            "/clang:-O3",
            "/clang:-ffp-contract=fast",
            "/clang:-funroll-loops",
        ]
        if MARCH_NATIVE:
            extra_compile_args += ["/clang:-march=native"]
        # Monkey-patch distutils' MSVC compiler to invoke clang-cl instead
        # of cl.exe. clang-cl is an MSVC-compatible Clang frontend (same
        # switches) — distutils sees it as just another `cl.exe`.
        import distutils._msvccompiler as _msvc  # noqa: PLC0415
        _orig_initialize = _msvc.MSVCCompiler.initialize
        def _patched_initialize(self, plat_name=None):
            _orig_initialize(self, plat_name)
            self.cc = "clang-cl.exe"
        _msvc.MSVCCompiler.initialize = _patched_initialize
else:
    extra_compile_args += [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-pthread",
        "-ffp-contract=fast",  # fuse mul+add -> FMA, matches numba LLVM emission
        "-funroll-loops",
    ]
    if MARCH_NATIVE:
        extra_compile_args += ["-march=native"]
    extra_link_args += ["-pthread"]
    if sys.platform == "darwin":
        extra_compile_args += ["-mmacosx-version-min=10.14"]


native_ext = Extension(
    "ncolor._backend._impl",
    sources=["cpp/binding.cpp"],
    include_dirs=[pybind11.get_include(), "cpp"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)


class build_ext(_build_ext):
    """build_ext + post-build SMT calibration.

    After the .so/.pyd is built, import :mod:`ncolor._backend._smt` from
    the build output and run a calibration on a 1024^2 mask (~50-300 ms).
    Result is written to ``platformdirs.user_cache_dir("ncolor") /
    smt_threads.json`` keyed by hostname + CPU model. Subsequent
    ``auto_threads()`` calls hit the cache in <1 ms.

    Always re-runs (even if cache exists) so install/rebuild gets fresh
    timings — useful if CPU/firmware/thermal-policy changed since last
    install. Skips silently if numpy isn't yet available (pip build
    isolation). Set the env var ``NCOLOR_NO_CALIBRATE=1`` to disable
    entirely (CI / cross-compilation).
    """

    def run(self):
        super().run()
        if os.environ.get("NCOLOR_NO_CALIBRATE"):
            return
        srcdir = os.path.dirname(os.path.abspath(__file__))
        # The freshly-built package lives under self.build_lib (when
        # building from a clean tree) or under src/ (for --inplace).
        candidates = [p for p in (os.path.join(srcdir, "src"), self.build_lib)
                      if p and p not in sys.path]
        for p in candidates:
            sys.path.insert(0, p)
        try:
            try:
                import numpy  # noqa: F401  — needed by calibrate()
            except ImportError:
                print("[ncolor] numpy not available at build time; SMT "
                      "calibration deferred to first import.")
                return
            try:
                from ncolor._backend import _smt
                _smt.calibrate(force=True, verbose=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ncolor] SMT calibration skipped ({exc!r}); "
                      "auto_threads() will fall back to physical core count.")
        finally:
            for p in candidates:
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass


setup(
    name="ncolor",
    license="BSD",
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="Label matrix 4-color graph coloring + Voronoi expansion (C++).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/ncolor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[native_ext],
    cmdclass={"build_ext": build_ext},
    use_scm_version=True,
    install_requires=install_deps,
    extras_require=extras_deps,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
