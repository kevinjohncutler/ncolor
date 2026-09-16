"""Load the compiled backend through a local cache on network filesystems.

Network builds remain supported. On macOS, a network volume's quarantine
mount flag can block loading a compiled extension even when the file has
no quarantine attribute. Some network filesystems also stall signature
validation; Windows network paths can reject library loading.

For detected remote paths, copy the extension to the local user cache and
remove the quarantine attribute from that deliberate local copy. Cache
keys include source modification time and size, so rebuilding selects a
fresh path rather than reusing the dynamic loader's failed-path state.
Local package installations load directly.
"""
from __future__ import annotations

import importlib.machinery
import os
import shutil
import subprocess
import sys
import tempfile
import time
from threading import Lock
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_IMPL_BASENAME = "_impl"
_FQ_NAME = "ncolor._backend._impl"


def _user_cache_dir() -> Path:
    """Per-OS cache directory. ``platformdirs`` is a hard runtime dep
    (declared in ``setup.py``'s ``install_requires``)."""
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("ncolor"))


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
        if Path(abs_path).is_relative_to(mount_point) and (
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


_COPY_LOCK = Lock()


def _copy_off_remote(src: Path, dst: Path) -> None:
    # Serialize writers in this process. Some remote filesystems expose a
    # transient empty view when several renames replace the same destination.
    with _COPY_LOCK:
        _publish_extension(src, dst)


def _publish_extension(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete binaries. Concurrent imports must never load a
    # destination while another process is still writing its contents.
    fd, name = tempfile.mkstemp(prefix=".ncolor-", dir=dst.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(src, temporary)
        temporary.chmod(0o755)
        if sys.platform == "darwin":
            subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", str(temporary)],
                check=False, stderr=subprocess.DEVNULL,
            )
        try:
            os.replace(temporary, dst)
        except (PermissionError, FileExistsError):
            # Another importer may have published first. Windows can lock
            # that file; some network filesystems reject replacing it.
            # Network metadata can lag the completed writes. Read through
            # file handles and compare contents instead of trusting stat sizes.
            identical = False
            for attempt in range(6):
                try:
                    with temporary.open("rb") as candidate, dst.open("rb") as published:
                        while True:
                            chunk = candidate.read(1024 * 1024)
                            if published.read(len(chunk) or 1) != chunk:
                                break
                            if not chunk:
                                identical = True
                                break
                except OSError:
                    pass
                if identical:
                    break
                # A concurrent network rename can briefly expose an empty
                # cached view. Wait for that publication, without replacing it.
                if attempt < 5:
                    time.sleep(0.05 * (2 ** attempt))
            if not identical:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _load_impl():
    src = _find_impl()
    if _on_remote_mount(src):
        local = _local_cache_path(src)
        if not local.exists() or local.stat().st_size != src.stat().st_size:
            _copy_off_remote(src, local)
        load_path = local
    else:
        load_path = src

    spec = spec_from_file_location(_FQ_NAME, str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build spec for {load_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[_FQ_NAME] = mod
    return mod


_impl = _load_impl()

# Re-export the public engine classes so ``from ncolor._backend import Solver``
# works for internal callers (the high-level wrappers in ncolor.color /
# ncolor.expand).
ExpandEngine = _impl.ExpandEngine
Solver = _impl.Solver

# Expose the calibration submodule.
from . import _smt  # noqa: E402


def _maybe_calibrate_on_first_import() -> None:
    """Run SMT calibration once per machine if no cache entry exists.

    pip's wheel install has no post-install hook (``setup.py``'s
    ``cmdclass`` only fires for source builds), so for users who install
    a pre-built wheel, the SMT calibration that source-build users get at
    install time has to happen at first import instead. ~50–300 ms hidden
    under the user's first ``import ncolor``; subsequent imports are
    instant (they hit the cached JSON file).

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

__all__ = ["ExpandEngine", "Solver", "_smt"]
