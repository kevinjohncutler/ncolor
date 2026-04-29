"""Loader for the ``_ncolor_cpp_proto_impl`` C++ extension.

This Python module is imported as ``ncolor_cpp_proto`` and re-exports the
classes from the underlying ``_ncolor_cpp_proto_impl`` shared object.
The indirection exists for one reason: macOS dyld hangs in
``JustInTimeLoader::withRegions`` when ``dlopen()``-ing a ``.so`` file from
an SMB-mounted share with the ``quarantine`` flag set (Apple's syspolicyd
calls ``fcntl()`` on the file for code-signature validation, and SMB hangs
on those calls).

To work around this, on first import we copy the ``.so`` to a unique
local-disk cache directory, strip ``com.apple.quarantine``, and load it
from there. The cache key is the source-side ``(mtime_ns, size)`` so that
rebuilds of the underlying .so produce a fresh local path (avoiding stale
dyld path caches that retain prior failed load state).

On non-macOS platforms — and on macOS when the source dir isn't on smbfs —
this fast-paths to the regular ``import _ncolor_cpp_proto_impl``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


def _is_smbfs(path: Path) -> bool:
    """True if `path` lives on a macOS smbfs mount."""
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
        rest = line.split(" on ", 1)[1]
        mount_point, opts = rest.split(" (", 1)
        if abs_path.startswith(mount_point.rstrip()) and "smbfs" in opts:
            return True
    return False


def _find_impl_so() -> Path:
    """Locate the compiled extension next to this .py file."""
    matches = sorted(_THIS_DIR.glob("_ncolor_cpp_proto_impl*.so"))
    matches += sorted(_THIS_DIR.glob("_ncolor_cpp_proto_impl*.pyd"))
    if not matches:
        raise ImportError(
            "_ncolor_cpp_proto_impl shared object not found in "
            f"{_THIS_DIR}; did the extension build succeed?"
        )
    return matches[0]


def _local_cache_path(src: Path) -> Path:
    """Cache path keyed by source mtime + size so rebuilds get a fresh dir."""
    st = src.stat()
    sig = f"{st.st_mtime_ns}_{st.st_size}"
    return Path.home() / ".cache" / "ncolor_cpp_proto" / "lib" / sig / src.name


def _copy_to_local_and_clean(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    dst.chmod(0o755)
    # Strip any quarantine xattr that copy may have inherited from the SMB
    # source. dyld's syspolicyd path hangs when this xattr is present.
    subprocess.run(
        ["xattr", "-d", "com.apple.quarantine", str(dst)],
        check=False, stderr=subprocess.DEVNULL,
    )


def _load_impl():
    src_so = _find_impl_so()
    if _is_smbfs(src_so):
        local_so = _local_cache_path(src_so)
        if not local_so.exists():
            _copy_to_local_and_clean(src_so, local_so)
        load_path = local_so
    else:
        load_path = src_so

    spec = spec_from_file_location("_ncolor_cpp_proto_impl", str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build spec for {load_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["_ncolor_cpp_proto_impl"] = mod
    return mod


_impl = _load_impl()

# Re-export the public API so ``import ncolor_cpp_proto as nc`` keeps
# ``nc.Solver(...)`` etc. working unchanged.
ConnectEngine = _impl.ConnectEngine
ExpandEngine = _impl.ExpandEngine
Solver = _impl.Solver

__all__ = ["ConnectEngine", "ExpandEngine", "Solver"]
