import numpy as np

from ._engines import _use


_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1


def _compact_wide_labels(labels):
    """Rewrite labels that do not fit int32 as dense int32 codes.

    The fallback for int64 / uint32 / uint64 / float input holding a
    value outside the int32 range, which the engine reports with
    ``OverflowError`` instead of wrapping (a wrapped label turns negative
    and the format pass then treats it as background, so cells silently
    vanish). Codes follow sorted-unique order, which is what the engine's
    own compaction yields for an input that fits: a value of 0 stays 0, a
    negative minimum becomes the background the way the min-shift rule
    treats it, and an all-positive input keeps every value as a cell.
    Floats are truncated toward zero first, matching the engine's cast.
    """
    arr = np.asarray(labels)
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError("label array contains NaN or inf")
        arr = np.trunc(arr)
    uniq, inverse = np.unique(arr, return_inverse=True)
    codes = np.asarray(inverse).reshape(arr.shape).astype(np.int32, copy=False)
    if uniq.size and uniq[0] > 0:
        codes += 1
    return codes


def _to_int32_labels(labels, allow_compact=True):
    """``astype(int32)`` that refuses to wrap.

    Narrow integer and bool inputs cast directly. For wide integer and
    float inputs holding a value that does not fit, the labels are
    compacted (see :func:`_compact_wide_labels`) when ``allow_compact``
    is set, and ``OverflowError`` is raised otherwise (for callers that
    must keep label identities). Always returns a fresh array the caller
    may modify.
    """
    arr = np.asarray(labels)
    kind, size = arr.dtype.kind, arr.dtype.itemsize
    if kind == "b" or (kind in "iu" and size < 4) or arr.dtype == np.int32:
        return arr.astype(np.int32, copy=True)
    if kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError("label array contains NaN or inf")
        arr = np.trunc(arr)
    if kind not in "iuf":
        raise TypeError(f"unsupported label dtype {arr.dtype}")
    if arr.size == 0 or (arr.min() >= _INT32_MIN and arr.max() <= _INT32_MAX):
        return arr.astype(np.int32, copy=True)
    if not allow_compact:
        raise OverflowError(
            "label values outside the int32 range; compact them first with "
            "ncolor.format_labels")
    return _compact_wide_labels(arr)


def format_labels(labels, clean=False, min_area=9, despur=False,
                  verbose=False, background=None, ignore=False,
                  first_seen=False, _engine=None):
    """Compact labels into background=0, cells 1..N.

    ``clean=True`` splits disjoint components per label and drops
    components below ``min_area``. ``despur=True`` additionally runs
    :func:`delete_spurs_labels` (label-aware C++ despur kernel) as a
    pre-pass to remove spur pixels and 1-voxel-thick interior bridges
    before component splitting.

    ``background`` selects the source value mapped to zero. By default,
    zero is background, or the minimum value if negative.

    ``ignore=True`` preserves source ``0`` as an ignore marker and ``1``
    as background, with cells numbered from 2. Removed components become
    background. An explicit background must be 1 in this mode.

    ``first_seen=True`` numbers compacted labels in input scan order
    instead of the default ascending-source order. ~2× slower; only
    needed if downstream code requires that exact ordering.
    """
    arr = np.ascontiguousarray(labels)
    ignore_mask = None
    zero_is_default = (background == 0 and arr.dtype.kind in "biuf" and
                       (not arr.size or arr.min() >= 0))
    if ignore or (background is not None and not zero_is_default):
        # Preserve source identities until the explicit markers are removed,
        # including markers outside int32 and negative foreground labels.
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported label dtype {arr.dtype}")
        if arr.dtype.kind == "f":
            if not np.isfinite(arr).all():
                raise ValueError("label array contains NaN or inf")
            arr = np.trunc(arr)
        if ignore:
            if background is not None and background != 1:
                raise ValueError("ignore=True requires background=1 or None")
            ignore_mask = arr == 0
            foreground = (arr != 1) & ~ignore_mask
        else:
            foreground = arr != background
        _, inverse = np.unique(arr[foreground], return_inverse=True)
        codes = np.zeros(arr.shape, dtype=np.int32)
        codes[foreground] = inverse + 1
        arr = codes

    with _use(_engine) as engine:
        try:
            labels, n_used = engine._expand.format_labels(
                arr, first_seen=bool(first_seen))
        except OverflowError:
            labels, n_used = engine._expand.format_labels(
                _compact_wide_labels(arr), first_seen=bool(first_seen))

        if clean and n_used:
            if despur:
                labels, _ = engine._expand.delete_spurs_labels(
                    labels, threshold=1, max_iters=20, remove_thin=True)
            components, n_total, sources = engine._expand.components_per_label(
                labels, conn=labels.ndim)
            areas = np.bincount(components.ravel(), minlength=n_total + 1)
            remap = np.arange(n_total + 1, dtype=np.int32)
            remap[areas < min_area] = 0
            remap[0] = 0
            if verbose:
                n_removed = int(np.count_nonzero(areas[1:] < min_area))
                print('Removed', n_removed, 'components with area less than', min_area)
                if np.unique(sources).size < n_total:
                    print('Warning - found mask with disjoint label.')
            # Clean output retains its established component scan order.
            labels, n_used = engine._expand.format_labels(
                remap[components], first_seen=True)

    if verbose:
        print('Formatted', n_used, 'labels')
    if ignore:
        labels += 1
        labels[ignore_mask] = 0
        n_used += 1
    # Keep the established default return type and generic-path downcast.
    if not clean and not ignore and background is None:
        return labels
    dtype = np.uint8 if n_used <= 255 else np.uint16 if n_used <= 65535 else np.uint32
    return labels.astype(dtype, copy=False)


def delete_spurs(arr, hole_threshold=5, *, mode="cardinal",
                 threshold=None, max_iter=-1, kind="auto",
                 max_iters=None, remove_thin=False, _engine=None):
    """N-D spur cleanup. Dispatches on input contents:

    * **Binary mask** (max value ≤ 1, or ``bool`` dtype) — fills small
      bg holes, then iteratively prunes pixels whose foreground-
      neighbor count falls below ``threshold``. ``hole_threshold`` /
      ``mode`` / ``threshold`` / ``max_iter`` apply.
    * **Label image** (multiple non-zero values) — runs the label-
      aware variant from ``delete_spurs_labels`` instead: zeros pixels
      whose count of face-adjacent **same-label** neighbors is ≤
      ``threshold`` (default 1). ``max_iters`` (alias ``max_iter``)
      bounds the loop. ``hole_threshold`` and ``mode`` are ignored.
      Returns ``(cleaned_labels, n_removed)`` for parity with the cpp
      binding.

    Pass ``kind='binary'`` or ``kind='labels'`` to force one path.

    ``hole_threshold`` (binary mode, default 5): bg components with
    pixel count ≤ this value get filled into the foreground before
    pruning. Components touching an image boundary are exterior and never
    filled. Pass 0 to skip hole filling entirely.

    ``mode`` (binary mode) — connectivity used by the endpoint check:

    * ``"cardinal"`` (default) — face neighbors only (2·ndim of them).
      Catches pixels sticking out of a flat boundary; matches the
      the external-spur rule. Aggressive; fewer iterations to converge.
    * ``"total"`` — full-diagonal (3^ndim − 1 neighbors). Preserves
      diagonally connected features better than face-only connectivity.
      At the default threshold, a straight 1-voxel-wide line in 3D
      still has only two neighbors and is pruned. Lower ``threshold``
      to retain such thin features.

    ``threshold`` — binary mode default is ``None`` → ``ndim`` (pixel
    is a spur if fg-neighbor count is in ``[1, threshold)``); label
    mode default is 1.

    ``max_iter`` (binary) / ``max_iters`` (label) — cap the iterative
    pruning loop. Binary default ``-1`` runs to convergence; label
    default 20 caps at 20 rounds. Explicit ``max_iters=-1`` runs label
    cleanup to convergence; zero performs no removal.

    ``remove_thin`` (label mode only, default ``False``) — also zero
    1-voxel-thick straight interior pixels on every pruning round. A pixel
    qualifies iff it has exactly two same-label full-connectivity
    neighbors that sit at opposite offsets, including N-D diagonals.
    Useful when expand_labels leaves 1-px bridges between cells that would otherwise need many
    iterations of end-peeling to clear.
    """
    if kind not in ("auto", "binary", "labels"):
        raise ValueError(
            f"kind must be 'auto', 'binary', or 'labels', got {kind!r}")
    arr = np.ascontiguousarray(arr)
    if kind == "auto":
        if arr.dtype == bool or int(arr.max() if arr.size else 0) <= 1:
            kind = "binary"
        else:
            kind = "labels"
    from ._backend import _impl as _b
    if kind == "binary":
        if mode not in ("cardinal", "total"):
            raise ValueError(
                f"mode must be 'cardinal' or 'total', got {mode!r}")
        if remove_thin:
            raise ValueError(
                "remove_thin only applies to kind='labels'")
        arr_u8 = (arr != 0).astype(np.uint8)
        conn_kind = 1 if mode == "cardinal" else arr_u8.ndim
        thr = int(threshold) if threshold is not None else -1
        return _b.delete_spurs(arr_u8, int(hole_threshold), int(conn_kind),
                                thr, int(max_iter))
    # kind == "labels". Spur removal keeps label identities, so a value
    # outside int32 is an error here rather than a reason to renumber.
    arr32 = arr if arr.dtype == np.int32 else _to_int32_labels(arr, allow_compact=False)
    thr = int(threshold) if threshold is not None else 1
    rounds = int(max_iters) if max_iters is not None else (
        20 if max_iter == -1 else int(max_iter))
    with _use(_engine) as engine:
        return engine._expand.delete_spurs_labels(
            arr32, threshold=thr, max_iters=rounds, remove_thin=bool(remove_thin))