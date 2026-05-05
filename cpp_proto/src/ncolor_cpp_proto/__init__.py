"""Loader for the ``_ncolor_cpp_proto_impl`` C++ extension.

This package exists as a thin Python wrapper around a single C++ extension
``_ncolor_cpp_proto_impl.<py-tag>.so`` (or ``.pyd`` on Windows). The wrapper
exists for one reason: ``dlopen()`` of a compiled extension hangs or fails
when the file lives on a network filesystem.

  * **macOS smbfs** — dyld calls ``fcntl()`` for code-signature validation
    and SMB hangs on those calls; ``dlopen`` blocks indefinitely in
    ``JustInTimeLoader::withRegions``.
  * **Windows UNC** — ``LoadLibrary`` raises *Access is denied* for some
    server configurations.

On first import, if the source ``.so``/``.pyd`` lives on a network mount,
we copy it to a unique local-disk cache directory and load it from there
instead. The cache key is the source-side ``(mtime_ns, size)`` so a rebuild
produces a fresh local path — dyld retains stale path-keyed state from
prior failed loads at the same path, so reusing the same path can still
hang. On macOS we also strip ``com.apple.quarantine`` (which ``cp`` from an
SMB mount inherits, even when the user has stripped it from the file).

On non-network mounts we fast-path to a direct ``importlib`` load.
"""
from __future__ import annotations

import importlib.machinery
import os
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_IMPL_BASENAME = "_ncolor_cpp_proto_impl"


def _user_cache_dir() -> Path:
    """Per-OS cache directory. ``platformdirs`` is a hard runtime dep
    (declared in ``setup.py``'s ``install_requires``)."""
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("ncolor_cpp_proto"))


_CACHE_ROOT = _user_cache_dir() / "lib"


def _on_remote_mount(path: Path) -> bool:
    """True if ``path`` lives on a network filesystem we know to be hostile
    to ``dlopen``: smbfs / nfs on POSIX, UNC paths on Windows."""
    if os.name == "nt":
        return path.is_absolute() and path.anchor.startswith("\\\\")
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.check_output(["mount"], text=True)
    except Exception:
        return False
    abs_path = str(path.resolve())
    for line in out.splitlines():
        if " on " not in line or " (" not in line:
            continue
        mount_point, opts = line.split(" on ", 1)[1].split(" (", 1)
        if abs_path.startswith(mount_point.rstrip()) and (
            "smbfs" in opts or "nfs" in opts or "afpfs" in opts
        ):
            return True
    return False


def _find_impl() -> Path:
    """Locate the compiled extension matching this Python's platform tag.

    On a NAS-shared package directory, .so files for *every* host that has
    built here may live alongside each other (e.g.,
    ``cpython-310-x86_64-linux-gnu.so`` and ``cpython-312-darwin.so``).
    Use ``importlib.machinery.EXTENSION_SUFFIXES`` so we only ever pick the
    one this interpreter can actually load.
    """
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _THIS_DIR / f"{_IMPL_BASENAME}{suffix}"
        if candidate.exists():
            return candidate
    raise ImportError(
        f"{_IMPL_BASENAME} extension matching this Python's platform tag "
        f"not found in {_THIS_DIR}; did the build succeed for "
        f"{sys.platform} {sys.implementation.name} {sys.version_info[:2]}?"
    )


def _local_cache_path(src: Path) -> Path:
    """Cache key on (mtime_ns, size) so rebuilds get a fresh local path."""
    st = src.stat()
    return _CACHE_ROOT / f"{st.st_mtime_ns}_{st.st_size}" / src.name


def _copy_off_remote(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    dst.chmod(0o755)
    if sys.platform == "darwin":
        # ``cp`` from a quarantined SMB mount inherits ``com.apple.quarantine``
        # on the destination (even though the source xattr is hidden by smbfs).
        # macOS Gatekeeper hangs on quarantined files; strip it.
        subprocess.run(
            ["xattr", "-d", "com.apple.quarantine", str(dst)],
            check=False, stderr=subprocess.DEVNULL,
        )


def _load_impl():
    src = _find_impl()
    if _on_remote_mount(src):
        local = _local_cache_path(src)
        if not local.exists():
            _copy_off_remote(src, local)
        load_path = local
    else:
        load_path = src

    spec = spec_from_file_location(_IMPL_BASENAME, str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build spec for {load_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[_IMPL_BASENAME] = mod
    return mod


_impl = _load_impl()

# Re-export the public API so ``import ncolor_cpp_proto as nc`` keeps
# ``nc.Solver(...)`` etc. working unchanged.
ConnectEngine = _impl.ConnectEngine
ExpandEngine = _impl.ExpandEngine
Solver = _impl.Solver

# Make the SMT calibration submodule discoverable via ``ncolor_cpp_proto._smt``.
from . import _smt  # noqa: E402


def _maybe_calibrate_on_first_import() -> None:
    """Run SMT calibration once per machine if no cache entry exists.

    pip's wheel install has no post-install hook (``setup.py``'s
    ``cmdclass`` only fires for source builds), so for users who install
    a pre-built wheel, the SMT calibration that source-build users get at
    install time has to happen at first import instead. ~50–300 ms hidden
    under the user's first ``import ncolor_cpp_proto``; subsequent imports
    are instant (they hit the cached JSON file).

    Skip with ``NCOLOR_NO_CALIBRATE=1`` (CI / Docker / cross-compile).
    Skip if numpy isn't yet importable (something has gone very wrong).
    Failures are non-fatal — ``auto_threads()`` falls back to physical
    core count.
    """
    if os.environ.get("NCOLOR_NO_CALIBRATE"):
        return
    cache = _smt._load_cache()
    if _smt._cache_key() in cache:
        return  # already calibrated on this host
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    try:
        _smt.calibrate(force=False, verbose=False)
    except Exception:
        pass  # non-fatal — auto_threads() falls back to physical cores


_maybe_calibrate_on_first_import()

__all__ = ["ConnectEngine", "ExpandEngine", "Solver", "_smt"]
