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
``cl.exe`` for ``clang-cl.exe`` — clang-cl's LLVM autovectorizer handles
the L1 inner loops MSVC's auto-vectorizer punts on.
"""
from __future__ import annotations

import glob
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

install_deps = [
    "numpy",
    "platformdirs",  # SMT calibration cache + native loader
]
extras_deps = {
    # Vector-geometry front end (ncolor.geo.label / .connect). Shapely
    # 2.0 is the floor: the vectorized STRtree.query + predicate API the
    # adjacency scan is built on landed there. GeoPandas is NOT required:
    # GeoDataFrames are handled by duck-typing, and geopandas itself
    # depends on shapely, so anyone passing one already has it.
    "geo": ["shapely>=2.0"],
}

extra_compile_args = []
extra_link_args = []
# Target CPU selection:
#   NCOLOR_MARCH_NATIVE=1 (default for source builds)  -> -march=native
#   NCOLOR_MARCH=<arch>   (wheel builds)                -> -march=<arch>
#   neither                                             -> compiler baseline
# -march=native picks AVX2/AVX-512 on Zen, NEON on Apple Silicon, etc.,
# but produces binaries that crash on older CPUs, so cibuildwheel turns
# it off. The x86_64 wheels instead get NCOLOR_MARCH=x86-64-v2 (SSE4.2,
# 2009+ CPUs, the same floor NumPy 2 requires), which turns on the SSE4.1
# lane multiply in expand.hpp; the plain SSE2 baseline still gets the
# 4-wide path through an emulated multiply. The arm64 NEON paths are
# gated on ``__aarch64__`` and stay on regardless of the march flag.
MARCH_NATIVE = os.environ.get("NCOLOR_MARCH_NATIVE", "1") == "1"
MARCH = os.environ.get("NCOLOR_MARCH", "").strip()


def _want_clang_cl():
    """clang-cl requested (NCOLOR_USE_CLANG_CL=1) and actually on PATH."""
    if os.environ.get("NCOLOR_USE_CLANG_CL") != "1":
        return False
    import shutil  # noqa: PLC0415
    if shutil.which("clang-cl") is None and shutil.which("clang-cl.exe") is None:
        print("[ncolor] NCOLOR_USE_CLANG_CL=1 but clang-cl is not on PATH; "
              "building with cl.exe instead.")
        return False
    return True


def _patch_msvc_to_clang_cl():
    """Make distutils' MSVC compiler class invoke clang-cl.exe.

    clang-cl is an MSVC-compatible Clang front end (same switches), so
    distutils sees it as just another ``cl.exe``. setuptools has moved
    the MSVC compiler class twice; try the current location first and
    the legacy one second.
    """
    candidates = (
        "setuptools._distutils.compilers.C.msvc",   # setuptools >= 74
        "distutils._msvccompiler",                  # older setuptools shim
    )
    last_exc = None
    for modname in candidates:
        try:
            import importlib  # noqa: PLC0415
            mod = importlib.import_module(modname)
            cls = getattr(mod, "Compiler", None) or getattr(mod, "MSVCCompiler")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        orig_initialize = cls.initialize

        def _patched_initialize(self, plat_name=None, _orig=orig_initialize):
            _orig(self, plat_name)
            self.cc = "clang-cl.exe"

        cls.initialize = _patched_initialize
        return
    raise RuntimeError(f"could not locate the distutils MSVC compiler class: {last_exc!r}")


USE_CLANG_CL = sys.platform == "win32" and _want_clang_cl()

if sys.platform == "win32":
    extra_compile_args += ["/std:c++17", "/O2", "/EHsc"]
    # NOMINMAX: prevent <windows.h> from defining `min` / `max` macros that
    #   collide with `std::min` / `std::max` throughout the engine
    #   (cc_label.hpp, chamfer.hpp, expand.hpp, expand_clean.hpp,
    #   format_labels.hpp all use std::min/std::max).
    # WIN32_LEAN_AND_MEAN: trim the windows.h include set; faster compile,
    #   smaller namespace pollution.
    extra_compile_args += ["/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
    if MARCH_NATIVE:
        # MSVC has no -march=native; AVX2 is the closest portable-enough
        # stand-in for a developer box (every x86 CPU since 2013).
        extra_compile_args += ["/arch:AVX2"]
    if USE_CLANG_CL:
        # clang-cl maps /O2 -> -O2; push to -O3 + the GCC-style flags via
        # /clang:. Without LTO the host MS link.exe is fine.
        extra_compile_args += [
            "/clang:-O3",
            "/clang:-ffp-contract=fast",
            "/clang:-funroll-loops",
        ]
        if MARCH_NATIVE:
            extra_compile_args += ["/clang:-march=native"]
        elif MARCH:
            extra_compile_args += [f"/clang:-march={MARCH}"]
        _patch_msvc_to_clang_cl()
else:
    extra_compile_args += [
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-pthread",
        "-ffp-contract=fast",  # fuse mul+add -> FMA, matches numba LLVM emission
        "-funroll-loops",
        # No debug info in the shared object. Python's own CFLAGS carry
        # -g, and on manylinux that shipped 25 MB of .debug_* sections in
        # a 27 MB .so (1.8 MB of code): a 6 MB wheel instead of 0.5 MB.
        "-g0",
    ]
    if MARCH_NATIVE:
        extra_compile_args += ["-march=native"]
    elif MARCH:
        extra_compile_args += [f"-march={MARCH}"]
    extra_link_args += ["-pthread"]
    if sys.platform == "darwin":
        extra_compile_args += ["-mmacosx-version-min=10.14"]
    elif sys.platform.startswith("linux"):
        # Drop the static symbol table too; the dynamic symbols the
        # loader needs are kept.
        extra_link_args += ["-Wl,--strip-all"]


native_ext = Extension(
    "ncolor._backend._impl",
    sources=["cpp/binding.cpp"],
    # The engine is header-only; without this list a header edit does
    # not trigger a rebuild of the single translation unit.
    depends=sorted(glob.glob("cpp/*.hpp")) + ["cpp/threadpool.h"],
    include_dirs=[pybind11.get_include(), "cpp"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)


def _branch_alignment_flag(compiler):
    """The flag that keeps x86 jumps inside 32-byte blocks, or None.

    Skylake-derived Intel cores (Skylake through Comet Lake, including the
    i9-9900K) carry a microcode fix for the JCC erratum that stops caching
    decoded instructions for any jump that crosses or ends on a 32-byte
    boundary. A hot loop that lands there runs from the legacy decoder, so
    an unrelated edit that shifts code layout can slow a kernel by 20%:
    a one-line change to the adjacency scan measured 0.79x on sparse
    images on an i9 and was faster everywhere once aligned. Clang spells
    the option as a driver flag and GCC passes it to the assembler; arm64,
    universal2 and older assemblers reject both, so the flag is used only
    where the real compiler accepts it for the real target.
    """
    import tempfile  # noqa: PLC0415
    for flag in ("-mbranches-within-32B-boundaries",
                 "-Wa,-mbranches-within-32B-boundaries"):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "probe.cpp")
            with open(src, "w") as fh:
                fh.write("int probe(int x) { return x + 1; }\n")
            # A rejected flag is the expected answer on most targets, so
            # keep the compiler's error message out of the build log.
            saved = os.dup(2)
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 2)
            try:
                compiler.compile([src], output_dir=tmp,
                                 extra_postargs=extra_compile_args + [flag])
            except Exception:  # noqa: BLE001
                continue
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            return flag
    return None


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

    def build_extensions(self):
        if self.compiler.compiler_type == "unix":
            flag = _branch_alignment_flag(self.compiler)
            if flag:
                for ext in self.extensions:
                    ext.extra_compile_args.append(flag)
        super().build_extensions()

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
