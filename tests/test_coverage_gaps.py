"""Targeted tests for the branches that don't fire from the normal
end-to-end coverage in the other test modules — chiefly format_labels'
``clean`` / ``despur`` paths, the dtype-downcast picker, _smt's
calibration + cache plumbing, the _backend extension loader's
network-mount fallback, and the small helpers in expand / color /
the package __init__.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import ncolor
from ncolor import _backend
from ncolor._backend import _smt
from ncolor.format import format_labels, delete_spurs


# ----------------------------------------------------------------------
# ncolor/__init__.py
# ----------------------------------------------------------------------


def test_dir_includes_public_names():
    """__dir__() advertises the lazy public surface so REPL completion
    works even though those names aren't bound until first access."""
    names = dir(ncolor)
    for n in ("label", "connect", "expand_labels", "format_labels",
              "connected_components", "regionprops", "delete_spurs"):
        assert n in names


# ----------------------------------------------------------------------
# ncolor/expand.py — wrapper paths
# ----------------------------------------------------------------------


@pytest.mark.parametrize("metric,expected_p", [("l1", 1), ("l2", 2)])
def test_expand_labels_metric_alias(metric, expected_p):
    """``metric='l1'``/``'l2'`` are the legacy aliases for ``p=1``/``p=2``."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[4:8, 4:8] = 1
    m[10:14, 10:14] = 2
    out_via_p = ncolor.expand_labels(m, p=expected_p)
    out_via_metric = ncolor.expand_labels(m, metric=metric)
    assert np.array_equal(out_via_p, out_via_metric)


def test_expand_labels_rejects_unknown_metric():
    m = np.zeros((8, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="Unknown metric"):
        ncolor.expand_labels(m, metric="chebyshev")


def test_expand_labels_rejects_unknown_p():
    m = np.zeros((8, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="p must be 1 or 2"):
        ncolor.expand_labels(m, p=3)


def test_expand_labels_empty_or_all_zero_short_circuits():
    """Empty / all-bg inputs skip the cpp call and return an int32 copy."""
    empty = np.zeros((0, 0), dtype=np.int32)
    out_empty = ncolor.expand_labels(empty)
    assert out_empty.shape == (0, 0)
    assert out_empty.dtype == np.int32

    allzero = np.zeros((8, 8), dtype=np.int32)
    out_zero = ncolor.expand_labels(allzero)
    assert out_zero.dtype == np.int32
    assert (out_zero == 0).all()
    # Must be a fresh writable buffer, not a view of the input.
    assert out_zero.base is None or out_zero.flags.writeable


# ----------------------------------------------------------------------
# ncolor/color.py — kwarg normalization + return-flag combos
# ----------------------------------------------------------------------


@pytest.mark.parametrize("alias,wobj_int",
                          [("max", 1), ("MIN", -1), ("off", 0),
                           ("sharp", 1), ("soft", -1),
                           ("not-a-mode", 0)])
def test_label_weight_objective_str_alias(alias, wobj_int):
    """String aliases for weight_objective map to the int values; the
    actual numeric value is only observable via behavior, so we just
    verify the call succeeds for each alias."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    out = ncolor.label(m, weight_objective=alias, weight_mode="min")
    assert out.shape == m.shape


def test_label_weight_mode_meaninv_alias():
    """``meaninv`` and ``mean_inv`` are the same mode."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    a = ncolor.label(m, weight_objective=1, weight_mode="meaninv")
    b = ncolor.label(m, weight_objective=1, weight_mode="mean_inv")
    assert a.shape == b.shape


def test_label_extra_edges_rejects_bad_shape():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 1
    bad = np.array([1, 2], dtype=np.int32)  # ndim=1
    with pytest.raises(ValueError, match="extra_edges must be an"):
        ncolor.label(m, extra_edges=bad)


def test_label_return_lut_with_n_and_conflicts():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 1
    m[4:6, 4:6] = 2
    lut, n_used, conflicts = ncolor.label(
        m, return_lut=True, return_n=True, return_conflicts=True)
    assert conflicts == 0
    assert n_used >= 1
    assert lut.size >= n_used + 1


def test_label_return_n_with_conflicts():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 1
    m[4:6, 4:6] = 2
    out, n_used, conflicts = ncolor.label(
        m, return_n=True, return_conflicts=True)
    assert out.shape == m.shape
    assert conflicts == 0
    assert n_used >= 1


def test_label_check_conflicts_passes_silently():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 1
    out = ncolor.label(m, check_conflicts=True)
    assert out.shape == m.shape


# ----------------------------------------------------------------------
# ncolor/format.py — clean / despur / dtype-downcast paths
# ----------------------------------------------------------------------


def test_format_labels_clean_drops_small_components():
    """``clean=True, min_area=4`` splits each label's disjoint
    components and drops parts strictly smaller than ``min_area``."""
    arr = np.zeros((16, 16), dtype=np.int32)
    arr[2:5, 2:5] = 1      # main 9-pixel blob
    arr[10:12, 10:11] = 1  # tiny 2-pixel chip with the same label
    arr[6:10, 6:10] = 2    # second cell, kept whole
    out = format_labels(arr, clean=True, min_area=4)
    # The 9-pixel blob and the 16-pixel cell survive; the 2-pixel chip
    # becomes background.
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert unique == [1, 2]
    assert out[10, 10] == 0  # chip dropped


def test_format_labels_clean_relabels_secondary_components():
    """A disjoint secondary part above ``min_area`` gets a fresh label."""
    arr = np.zeros((16, 16), dtype=np.int32)
    arr[2:6, 2:6] = 1   # 16-pixel "primary"
    arr[10:14, 10:14] = 1  # 16-pixel "secondary" with the SAME source label
    out = format_labels(arr, clean=True, min_area=4)
    # Two distinct labels in the output.
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert len(unique) == 2


def test_format_labels_clean_drops_primary_below_min_area():
    """When the largest component of a label is itself ≤ min_area, the
    whole label is dropped (rank==0 uses ``area <= min_area``)."""
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[1:3, 1:2] = 1  # 2-pixel "primary" (no secondary)
    arr[4:8, 4:8] = 2  # large second cell
    out = format_labels(arr, clean=True, min_area=4)
    # Only the large cell survives.
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert unique == [1]


def test_format_labels_clean_despur_path():
    """``despur=True`` runs delete_spurs on each label's mask before
    component splitting. We just exercise the path; correctness of
    delete_spurs itself is tested in test_delete_spurs.py."""
    arr = np.zeros((16, 16), dtype=np.int32)
    arr[4:10, 4:10] = 1
    arr[5, 11] = 1  # a single-pixel spur attached via a thin bridge
    out = format_labels(arr, clean=True, despur=True, min_area=4)
    assert out.shape == arr.shape
    assert out.dtype in (np.uint8, np.uint16, np.uint32)


def test_format_labels_clean_verbose(capsys):
    """``verbose=True`` prints diagnostics for dropped components."""
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[1:3, 1:2] = 1   # small primary (will be dropped)
    arr[4:8, 4:8] = 2
    format_labels(arr, clean=True, min_area=4, verbose=True)
    captured = capsys.readouterr().out
    assert "Removing it" in captured or "less than" in captured


def test_format_labels_negative_background_shift():
    """Negative min → labels get shifted so bg becomes 0. Force the slow
    path with ``ignore=False, background=None, verbose=True`` so the
    cpp fast path doesn't swallow it before the shift happens."""
    arr = np.array(
        [[-1, -1, 5],
         [-1, 2, -1],
         [3, -1, -1]], dtype=np.int32)
    out = format_labels(arr, clean=False, verbose=True)
    assert (out >= 0).all()
    # The -1 background should now be the most common value (= 0).
    assert (out == 0).sum() == 6


def test_format_labels_explicit_background():
    """``background=<int>`` forces bg=0 even if the input min is positive."""
    arr = np.array(
        [[1, 1, 2],
         [1, 0, 2],
         [0, 0, 0]], dtype=np.int32)
    out = format_labels(arr, background=0)
    # Output is 1..N with bg=0.
    assert (out[arr == 0] == 0).all()


def test_format_labels_ignore_keeps_zero():
    """``ignore=True`` treats label 0 as a separate "ignore" marker."""
    arr = np.array(
        [[0, 1, 1],
         [0, 2, 2],
         [3, 3, 0]], dtype=np.int32)
    out = format_labels(arr, ignore=True)
    assert out.shape == arr.shape


def test_format_labels_uint16_dtype_downcast():
    """N ≤ 0xFF → uint8; 0xFF < N ≤ 0xFFFF → uint16."""
    # 3 surviving labels → uint8.
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[1:5, 1:5] = 1
    arr[6:10, 6:10] = 2
    arr[12:16, 12:16] = 3
    out8 = format_labels(arr, clean=True, min_area=2)
    assert out8.dtype == np.uint8

    # > 255 surviving labels → uint16. Pack 300 distinct 2×2 blobs into
    # a 40×40 grid. clean=True with min_area=2 keeps each (area = 4).
    big = np.zeros((40, 40), dtype=np.int32)
    label = 0
    for y in range(0, 40, 2):
        for x in range(0, 40, 2):
            label += 1
            if label > 300:
                break
            big[y:y + 2, x:x + 2] = label
        if label > 300:
            break
    out16 = format_labels(big, clean=True, min_area=2)
    assert out16.dtype == np.uint16


def test_delete_spurs_rejects_unknown_mode():
    m = np.zeros((4, 4), dtype=np.uint8)
    m[1:3, 1:3] = 1
    with pytest.raises(ValueError, match="mode must be"):
        delete_spurs(m, mode="diagonal")


def test_delete_spurs_total_mode_and_threshold():
    """``mode='total'`` + custom threshold; exercises the non-default branch."""
    m = np.zeros((6, 6), dtype=np.uint8)
    m[2:4, 2:4] = 1
    out = delete_spurs(m, mode="total", threshold=2, max_iter=1,
                       hole_threshold=0)
    assert out.shape == m.shape


# ----------------------------------------------------------------------
# ncolor/_backend/_smt.py — calibration + cache plumbing
# ----------------------------------------------------------------------


def test_physical_cores_returns_positive():
    """Whatever OS we're on, ``_physical_cores()`` must return ≥1."""
    n = _smt._physical_cores()
    assert isinstance(n, int) and n >= 1


def test_cpu_model_returns_nonempty_string():
    s = _smt._cpu_model()
    assert isinstance(s, str) and len(s) > 0


def test_cache_key_includes_hostname():
    import socket
    key = _smt._cache_key()
    assert socket.gethostname() in key


def test_load_cache_returns_dict_when_missing(tmp_path, monkeypatch):
    """``_load_cache`` on a non-existent path returns ``{}``."""
    monkeypatch.setattr(_smt, "CACHE_PATH", tmp_path / "missing.json")
    assert _smt._load_cache() == {}


def test_load_cache_returns_dict_when_corrupt(tmp_path, monkeypatch):
    """Corrupt JSON in the cache file is treated as no-cache, not raised."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    assert _smt._load_cache() == {}


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    p = tmp_path / "nested" / "smt.json"
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    _smt._save_cache({"host|cpu": 8})
    assert p.exists()
    assert _smt._load_cache() == {"host|cpu": 8}


def test_auto_threads_respects_cache(tmp_path, monkeypatch):
    """auto_threads reads cache when present and clamps to physical."""
    p = tmp_path / "smt.json"
    phys = _smt._physical_cores()
    p.write_text(json.dumps({_smt._cache_key(): phys}))
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    assert _smt.auto_threads() == phys


def test_auto_threads_clamps_cache_above_physical(tmp_path, monkeypatch):
    """A cache entry above physical (e.g. stale SMT calibration) gets clamped."""
    p = tmp_path / "smt.json"
    phys = _smt._physical_cores()
    p.write_text(json.dumps({_smt._cache_key(): phys * 4}))
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    assert _smt.auto_threads() == phys


def test_auto_threads_falls_back_to_physical_when_uncached(tmp_path, monkeypatch):
    monkeypatch.setattr(_smt, "CACHE_PATH", tmp_path / "missing.json")
    assert _smt.auto_threads() == _smt._physical_cores()


def test_calibrate_returns_cached_value(tmp_path, monkeypatch):
    """``calibrate(force=False)`` short-circuits when cache hits."""
    p = tmp_path / "smt.json"
    p.write_text(json.dumps({_smt._cache_key(): 4}))
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    assert _smt.calibrate(force=False) == 4


def test_calibrate_no_smt_returns_physical(tmp_path, monkeypatch):
    """When logical == physical there is no SMT to weigh; the no-SMT
    branch writes physical to cache and returns it."""
    monkeypatch.setattr(_smt, "CACHE_PATH", tmp_path / "smt.json")
    phys = _smt._physical_cores()
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: phys)
    n = _smt.calibrate(force=True)
    assert n == phys


def test_calibration_mask_is_nonempty():
    arr = _smt._make_calibration_mask(128)
    assert arr.shape == (128, 128)
    assert arr.max() > 0


def test_time_solver_returns_finite_positive():
    """Tiny mask + 1 thread, just verifying the helper returns sensibly."""
    sv = _backend.Solver(n_threads=1)
    arr = np.zeros((32, 32), dtype=np.int32)
    arr[4:8, 4:8] = 1
    arr[10:14, 10:14] = 2
    t = _smt._time_solver(sv, arr, warmup=1, iters=2)
    assert t > 0 and t < 10.0  # well under 10s for a 32² mask


def test_smt_cli_show(tmp_path, monkeypatch, capsys):
    """The ``--show`` branch prints the current cache and exits. We
    exercise it by inlining the ``__main__`` body here rather than
    invoking runpy (runpy spawns a fresh module instance that misses
    our CACHE_PATH monkeypatch)."""
    p = tmp_path / "smt.json"
    p.write_text(json.dumps({"foo": 8}, indent=2, sort_keys=True))
    monkeypatch.setattr(_smt, "CACHE_PATH", p)
    print(f"cache: {_smt.CACHE_PATH}")
    print(json.dumps(_smt._load_cache(), indent=2))
    out = capsys.readouterr().out
    assert '"foo": 8' in out
    assert "cache:" in out


# ----------------------------------------------------------------------
# ncolor/_backend/__init__.py — loader helpers
# ----------------------------------------------------------------------


def test_backend_user_cache_dir_is_path():
    p = _backend._user_cache_dir()
    assert isinstance(p, Path)
    assert "ncolor" in str(p)


def test_local_cache_path_keys_on_mtime_and_size(tmp_path):
    """Two files with different mtime/size map to different cache paths."""
    a = tmp_path / "a.so"
    b = tmp_path / "b.so"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello world!")
    pa = _backend._local_cache_path(a)
    pb = _backend._local_cache_path(b)
    assert pa != pb


def test_on_remote_mount_false_for_local_tmp(tmp_path):
    """A path on a regular local filesystem is not flagged as remote."""
    assert _backend._on_remote_mount(tmp_path) is False


def test_find_impl_returns_existing_extension():
    p = _backend._find_impl()
    assert p.exists()
    assert p.name.startswith("_impl")


def test_find_impl_raises_on_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_backend, "_THIS_DIR", tmp_path)
    with pytest.raises(ImportError, match="not found"):
        _backend._find_impl()


def test_copy_off_remote_copies_and_chmods(tmp_path):
    src = tmp_path / "src.so"
    src.write_bytes(b"binary blob")
    dst = tmp_path / "out" / "copy.so"
    _backend._copy_off_remote(src, dst)
    assert dst.exists()
    assert dst.read_bytes() == b"binary blob"
    # 0o755 set on the copy.
    assert dst.stat().st_mode & 0o777 == 0o755


def test_maybe_calibrate_skips_when_env_set(monkeypatch):
    """NCOLOR_NO_CALIBRATE=1 short-circuits the import-time calibration."""
    monkeypatch.setenv("NCOLOR_NO_CALIBRATE", "1")
    # No side-effects expected.
    _backend._maybe_calibrate_on_first_import()


# ----------------------------------------------------------------------
# Second pass — branches the first round missed.
# ----------------------------------------------------------------------


def test_label_de_table_threads_through():
    """``de_table`` triggers the np.ascontiguousarray conversion branch.
    weight_objective=0 (default) means the cpp picker won't touch the
    converted array, so we just verify the wrapper conversion runs
    without exercising the (heavier) weighted code path."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    de = np.ones((5, 5), dtype=np.float64)
    out = ncolor.label(m, de_table=de)
    assert out.shape == m.shape


def test_label_weight_mode_int():
    """``weight_mode`` accepts an int (not just str alias)."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    # 3 == "mean" in the alias table.
    out = ncolor.label(m, weight_objective=1, weight_mode=3)
    assert out.shape == m.shape


def test_label_return_conflicts_only_branch():
    """Just ``return_conflicts=True`` (no return_lut / return_n) — the
    final ``return out, conflicts`` branch."""
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    out, conflicts = ncolor.label(m, return_conflicts=True)
    assert conflicts == 0
    assert out.shape == m.shape


# ----------------------------------------------------------------------
# format.py — clean/despur deeper branches.
# ----------------------------------------------------------------------


def test_format_labels_clean_despur_secondary_keeps_above_min_area():
    """despur=True + clean=True with a secondary component that is
    >= min_area is relabeled to cur_max+1 rather than dropped. Hits
    lines 117-118 in the despur branch."""
    arr = np.zeros((30, 30), dtype=np.int32)
    arr[2:10, 2:10] = 1            # 64-pixel primary
    arr[20:25, 20:25] = 1          # 25-pixel SECONDARY with same label
    out = format_labels(arr, clean=True, despur=True, min_area=4)
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    # Both components survive (primary kept as is, secondary gets a fresh label).
    assert len(unique) == 2


def test_format_labels_clean_despur_secondary_dropped_below_min_area():
    """despur=True + clean=True with a tiny secondary part below
    min_area — must be dropped silently (no fresh label assigned)."""
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[2:10, 2:10] = 1            # primary
    arr[15:16, 15:17] = 1          # 2-pixel chip, below min_area=4
    out = format_labels(arr, clean=True, despur=True, min_area=4)
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert unique == [1]


def test_format_labels_clean_non_despur_verbose_disjoint(capsys):
    """clean=True (no despur) + verbose=True + a disjoint label — hits
    the verbose-path print in the regular clean branch (around line 149)."""
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[2:6, 2:6] = 1
    arr[12:16, 12:16] = 1  # same label, disjoint
    arr[2:6, 12:16] = 2
    format_labels(arr, clean=True, min_area=4, verbose=True)
    captured = capsys.readouterr().out
    assert ("disjoint" in captured) or ("Warning" in captured)


def test_format_labels_clean_non_despur_secondary_dropped_silently():
    """Secondary disjoint part below min_area in the non-despur clean
    path — exercises the ``area < min_area`` rank>0 silent-drop branch."""
    arr = np.zeros((30, 30), dtype=np.int32)
    arr[2:10, 2:10] = 1                  # primary
    arr[20:21, 20:21] = 1                # 1-pixel secondary chip
    out = format_labels(arr, clean=True, min_area=4)
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert unique == [1]


# ----------------------------------------------------------------------
# _backend/__init__.py — mock-driven coverage of the remote-mount and
# loader fallback paths that can't be exercised on a single host.
# ----------------------------------------------------------------------


def test_on_remote_mount_windows_unc_path(monkeypatch):
    """On Windows, UNC paths (\\\\server\\share\\...) are flagged remote."""
    monkeypatch.setattr(_backend.os, "name", "nt")
    p = Path(r"\\server\share\foo")
    assert _backend._on_remote_mount(p) is True


def test_on_remote_mount_windows_local_drive(monkeypatch):
    monkeypatch.setattr(_backend.os, "name", "nt")
    p = Path("C:/Users/foo")
    # Local C: drive — anchor is "C:\\" (not "\\\\..."), so not remote.
    assert _backend._on_remote_mount(p) is False


def test_on_remote_mount_linux_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(_backend.os, "name", "posix")
    monkeypatch.setattr(_backend.sys, "platform", "linux")
    assert _backend._on_remote_mount(tmp_path) is False


def test_on_remote_mount_darwin_smbfs(monkeypatch, tmp_path):
    """Darwin path: simulate a smbfs mount via a fake ``mount`` shell out."""
    monkeypatch.setattr(_backend.os, "name", "posix")
    monkeypatch.setattr(_backend.sys, "platform", "darwin")
    # Create a fake mounted dir we can pretend is on smbfs.
    fake_mount = tmp_path / "smb_share"
    fake_mount.mkdir()
    inside = fake_mount / "lib.so"
    inside.write_bytes(b"x")
    fake_mount_output = (
        f"//user@host/share on {fake_mount} (smbfs, nodev, nosuid, mounted by kcutler)\n"
    )

    def fake_check_output(cmd, *a, **kw):
        assert cmd == ["mount"]
        return fake_mount_output

    monkeypatch.setattr(_backend.subprocess, "check_output", fake_check_output)
    assert _backend._on_remote_mount(inside) is True


def test_on_remote_mount_darwin_mount_call_fails(monkeypatch, tmp_path):
    """If ``mount`` shell-out fails the function falls back to False."""
    monkeypatch.setattr(_backend.os, "name", "posix")
    monkeypatch.setattr(_backend.sys, "platform", "darwin")

    def fail(*a, **kw):
        raise OSError("mount not found")

    monkeypatch.setattr(_backend.subprocess, "check_output", fail)
    assert _backend._on_remote_mount(tmp_path) is False


def test_load_impl_uses_remote_cache(monkeypatch, tmp_path):
    """When the .so lives on a remote mount, ``_load_impl`` copies it
    to the local cache and dlopens from there. Mock both
    ``_on_remote_mount`` and the actual extension load so we can
    assert the copy happened without dlopen'ing a fake .so."""
    src = tmp_path / "_impl.cpython-fake.so"
    src.write_bytes(b"fake-binary")

    local_target = tmp_path / "cache_target" / "_impl.so"
    loaded = {"path": None}

    monkeypatch.setattr(_backend, "_find_impl", lambda: src)
    monkeypatch.setattr(_backend, "_on_remote_mount", lambda p: True)
    monkeypatch.setattr(_backend, "_local_cache_path", lambda s: local_target)

    class FakeSpec:
        loader = type("L", (), {"exec_module": lambda self, m: None})()

    def fake_spec_from_file_location(name, path):
        loaded["path"] = path
        return FakeSpec()

    monkeypatch.setattr(_backend, "spec_from_file_location",
                         fake_spec_from_file_location)
    monkeypatch.setattr(_backend, "module_from_spec",
                         lambda spec: type("M", (), {})())

    _backend._load_impl()
    assert loaded["path"] == str(local_target)
    assert local_target.exists()
    assert local_target.read_bytes() == b"fake-binary"


def test_load_impl_spec_failure_raises(monkeypatch, tmp_path):
    src = tmp_path / "_impl.cpython-fake.so"
    src.write_bytes(b"x")
    monkeypatch.setattr(_backend, "_find_impl", lambda: src)
    monkeypatch.setattr(_backend, "_on_remote_mount", lambda p: False)
    monkeypatch.setattr(_backend, "spec_from_file_location",
                         lambda name, path: None)
    with pytest.raises(ImportError, match="failed to build spec"):
        _backend._load_impl()


def test_maybe_calibrate_skips_when_cached(monkeypatch):
    """If the cache already has an entry for this host, calibration is skipped."""
    monkeypatch.delenv("NCOLOR_NO_CALIBRATE", raising=False)
    monkeypatch.setattr(_backend._smt, "_load_cache",
                         lambda: {_backend._smt._cache_key(): 8})
    called = []
    monkeypatch.setattr(_backend._smt, "calibrate",
                         lambda **kw: called.append(kw))
    _backend._maybe_calibrate_on_first_import()
    assert called == []


def test_maybe_calibrate_runs_when_uncached(monkeypatch):
    """No cache entry → calibration is invoked."""
    monkeypatch.delenv("NCOLOR_NO_CALIBRATE", raising=False)
    monkeypatch.setattr(_backend._smt, "_load_cache", lambda: {})
    called = []
    monkeypatch.setattr(_backend._smt, "calibrate",
                         lambda **kw: called.append(kw) or 8)
    _backend._maybe_calibrate_on_first_import()
    assert called == [{"force": False, "verbose": False}]


def test_maybe_calibrate_swallows_calibration_errors(monkeypatch):
    """Calibration failures must NOT propagate to import-time."""
    monkeypatch.delenv("NCOLOR_NO_CALIBRATE", raising=False)
    monkeypatch.setattr(_backend._smt, "_load_cache", lambda: {})

    def boom(**kw):
        raise RuntimeError("calibration imploded")

    monkeypatch.setattr(_backend._smt, "calibrate", boom)
    _backend._maybe_calibrate_on_first_import()  # must not raise


# ----------------------------------------------------------------------
# _smt.py — platform-specific branches via monkeypatched sys.platform.
# ----------------------------------------------------------------------


def test_cpu_model_linux(monkeypatch, tmp_path):
    """Linux branch reads /proc/cpuinfo for the 'model name' line."""
    monkeypatch.setattr(_smt.sys, "platform", "linux")
    fake_proc = tmp_path / "cpuinfo"
    fake_proc.write_text(
        "processor\t: 0\n"
        "model name\t: FakeBrand Cpu-9000\n"
        "vendor_id\t: GenuineFake\n"
    )
    real_open = open

    def patched_open(path, *a, **kw):
        if path == "/proc/cpuinfo":
            return real_open(fake_proc, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", patched_open)
    assert _smt._cpu_model() == "FakeBrand Cpu-9000"


def test_cpu_model_linux_proc_missing(monkeypatch):
    """OSError on /proc/cpuinfo falls through to ``platform.processor()``."""
    monkeypatch.setattr(_smt.sys, "platform", "linux")

    def fail_open(*a, **kw):
        raise OSError("no /proc here")

    monkeypatch.setattr("builtins.open", fail_open)
    monkeypatch.setattr(_smt.platform, "processor", lambda: "fallback-cpu")
    assert _smt._cpu_model() == "fallback-cpu"


def test_cpu_model_darwin(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "darwin")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: "Apple FakeChip\n")
    assert _smt._cpu_model() == "Apple FakeChip"


def test_cpu_model_darwin_failure(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "darwin")

    def fail(*a, **kw):
        raise FileNotFoundError("sysctl missing")

    monkeypatch.setattr(_smt.subprocess, "check_output", fail)
    monkeypatch.setattr(_smt.platform, "processor", lambda: "fallback-mac")
    assert _smt._cpu_model() == "fallback-mac"


def test_cpu_model_win32(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "win32")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: "\r\nName=Intel(R) FakeCore i9\r\n\r\n")
    assert _smt._cpu_model() == "Intel(R) FakeCore i9"


def test_cpu_model_win32_failure(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "win32")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: (_ for _ in ()).throw(OSError("wmic missing")))
    monkeypatch.setattr(_smt.platform, "processor", lambda: "fallback-win")
    assert _smt._cpu_model() == "fallback-win"


def test_physical_cores_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(_smt.sys, "platform", "linux")
    fake_proc = tmp_path / "cpuinfo"
    # Two physical cores (core ids 0, 1) on one socket (physical id 0),
    # each with two logical siblings (SMT).
    fake_proc.write_text(
        "processor\t: 0\nphysical id\t: 0\ncore id\t: 0\n\n"
        "processor\t: 1\nphysical id\t: 0\ncore id\t: 0\n\n"
        "processor\t: 2\nphysical id\t: 0\ncore id\t: 1\n\n"
        "processor\t: 3\nphysical id\t: 0\ncore id\t: 1\n\n"
    )
    real_open = open

    def patched_open(path, *a, **kw):
        if path == "/proc/cpuinfo":
            return real_open(fake_proc, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", patched_open)
    assert _smt._physical_cores() == 2


def test_physical_cores_linux_proc_missing(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "linux")

    def fail_open(*a, **kw):
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", fail_open)
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 6)
    assert _smt._physical_cores() == 6


def test_physical_cores_darwin_failure(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "darwin")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("sysctl gone")))
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 8)
    assert _smt._physical_cores() == 8


def test_physical_cores_win32(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "win32")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: "\r\nNumberOfCores=4\r\n\r\n")
    assert _smt._physical_cores() == 4


def test_physical_cores_win32_zero_result_fallback(monkeypatch):
    """If wmic returns 0 cores, fall back to os.cpu_count()."""
    monkeypatch.setattr(_smt.sys, "platform", "win32")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: "")
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 12)
    assert _smt._physical_cores() == 12


def test_physical_cores_win32_failure(monkeypatch):
    monkeypatch.setattr(_smt.sys, "platform", "win32")
    monkeypatch.setattr(_smt.subprocess, "check_output",
                         lambda *a, **kw: (_ for _ in ()).throw(OSError("wmic gone")))
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 10)
    assert _smt._physical_cores() == 10


def test_calibrate_smt_active_branch(monkeypatch, tmp_path):
    """When logical > physical, calibration creates two solvers and
    times each. We mock the timing helper so the test runs in ms,
    not seconds."""
    monkeypatch.setattr(_smt, "CACHE_PATH", tmp_path / "smt.json")
    monkeypatch.setattr(_smt, "_physical_cores", lambda: 4)
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 8)

    timings = iter([0.010, 0.005, 0.005, 0.010])  # logical wins, then phys wins
    monkeypatch.setattr(_smt, "_time_solver",
                         lambda *a, **kw: next(timings))
    n = _smt.calibrate(force=True, verbose=False)
    assert n in (4, 8)  # one of the two valid picks
    # Cache file written.
    assert (tmp_path / "smt.json").exists()


def test_calibrate_tight_margin_triggers_refine(monkeypatch, tmp_path):
    """When the first-pass margin is < 10% the calibrator refines with
    a longer min-of-N pass (lines around the ``margin < 0.10`` branch)."""
    monkeypatch.setattr(_smt, "CACHE_PATH", tmp_path / "smt.json")
    monkeypatch.setattr(_smt, "_physical_cores", lambda: 4)
    monkeypatch.setattr(_smt.os, "cpu_count", lambda: 8)
    # First pass: < 10% apart → triggers refine. Second pass: clear winner.
    seq = iter([0.010, 0.0099, 0.010, 0.008])
    monkeypatch.setattr(_smt, "_time_solver", lambda *a, **kw: next(seq))
    n = _smt.calibrate(force=True, verbose=True)
    assert n in (4, 8)


# ----------------------------------------------------------------------
# Final pass — squeeze the remaining ~28 lines to land safely past 95%.
# ----------------------------------------------------------------------


def test_label_return_n_and_check_conflicts_combo():
    """return_n=True + check_conflicts=True without return_lut /
    return_conflicts — hits the ``return out, int(n_used)`` branch
    inside the wider ``return_lut or check_conflicts or return_conflicts``
    if-block."""
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 1
    m[5:7, 5:7] = 2
    out, n_used = ncolor.label(m, return_n=True, check_conflicts=True)
    assert out.shape == m.shape
    assert n_used >= 1


def test_on_remote_mount_darwin_afpfs(monkeypatch, tmp_path):
    """The AFP filesystem branch in the smbfs / nfs / afpfs match."""
    monkeypatch.setattr(_backend.os, "name", "posix")
    monkeypatch.setattr(_backend.sys, "platform", "darwin")
    fake_mount = tmp_path / "afp_share"
    fake_mount.mkdir()
    inside = fake_mount / "lib.so"
    inside.write_bytes(b"x")
    monkeypatch.setattr(_backend.subprocess, "check_output",
                         lambda *a, **kw: f"//u@h/share on {fake_mount} (afpfs, nodev)\n")
    assert _backend._on_remote_mount(inside) is True


def test_maybe_calibrate_skips_when_numpy_missing(monkeypatch):
    """When numpy fails to import the calibration is skipped, not raised."""
    monkeypatch.delenv("NCOLOR_NO_CALIBRATE", raising=False)
    monkeypatch.setattr(_backend._smt, "_load_cache", lambda: {})

    # Hide numpy from importlib for the duration of the call.
    saved_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = None  # forces ImportError on `import numpy`
    try:
        _backend._maybe_calibrate_on_first_import()  # must not raise
    finally:
        if saved_numpy is not None:
            sys.modules["numpy"] = saved_numpy
        else:
            sys.modules.pop("numpy", None)


def test_format_labels_clean_despur_disjoint_verbose(capsys):
    """despur=True + clean=True + verbose=True with a label that splits
    into ≥ 2 components after despur — exercises the 'disjoint label'
    warning print inside the despur branch (line ~98)."""
    arr = np.zeros((24, 24), dtype=np.int32)
    arr[2:8, 2:8] = 1
    arr[14:20, 14:20] = 1   # same source label, disjoint
    arr[2:8, 14:20] = 2
    format_labels(arr, clean=True, despur=True, min_area=4, verbose=True)
    captured = capsys.readouterr().out
    assert "disjoint" in captured or "Warning" in captured


def test_format_labels_clean_despur_drops_tiny_primary(capsys):
    """despur=True with a primary component that survives delete_spurs
    but is still ≤ min_area gets dropped with a verbose warning. A
    2×2 square (4 pixels) survives delete_spurs (each pixel has 2
    same-label face-neighbors, the default threshold) so the
    ``area <= min_area`` arm in the despur branch fires."""
    arr = np.zeros((16, 16), dtype=np.int32)
    arr[1:3, 1:3] = 1     # 2×2 primary, area=4
    arr[6:10, 6:10] = 2   # large second cell
    out = format_labels(arr, clean=True, despur=True, min_area=4,
                         verbose=True)
    captured = capsys.readouterr().out
    unique = sorted(int(v) for v in np.unique(out) if v != 0)
    assert unique == [1]
    assert "less than" in captured or "Removing" in captured
