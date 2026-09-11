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

    ``ignore=True`` keeps ``0`` as an "ignore" marker and treats ``1``
    as background (the input min-shift is skipped).

    ``first_seen=True`` numbers compacted labels in input scan order
    instead of the default ascending-source order. ~2× slower; only
    needed if downstream code requires that exact ordering.
    """
    # Default-shape fast path: cpp engine does the dtype cast + renumber
    # under one GIL release. The generic path below has to do its own
    # min-shift / sign handling first, so it can't share this short-cut.
    if (not clean and not ignore and background is None and not verbose):
        arr = np.ascontiguousarray(labels)
        with _use(_engine) as engine:
            eng = engine._expand
            try:
                out, _n = eng.format_labels(arr, first_seen=bool(first_seen))
            except OverflowError:
                # A wide-dtype value outside int32: compact in numpy,
                # then let the engine renumber that.
                out, _n = eng.format_labels(_compact_wide_labels(arr),
                                            first_seen=bool(first_seen))
        return out

    # Cellpose stores labels inside float arrays; cast back to int.
    # Some segmenters use -1 as background, so use a signed dtype here.
    # Values outside int32 are compacted rather than wrapped.
    labels = _to_int32_labels(labels)
    if background is None:
        # Min-shift only when the min is negative; otherwise the smallest
        # cell would be absorbed into the background.
        m = int(np.min(labels))
        background = m if m < 0 else 0
    else:
        background = 0

    if not ignore:
        if verbose:
            print('minimum value is {}, shifting to 0'.format(background))
        if background != 0:
            labels -= background
            background = 0
    labels = labels.astype('uint32')

    if clean:
        from .color import regionprops as _regionprops
        from ._backend import _impl as _b

        ndim = labels.ndim
        # uint32 → int32 by view (zero-copy, same byte width). Safe
        # because no realistic label image has > 2^31 distinct values;
        # the upper bit is always clear so reinterpretation is identical.
        labels_i32 = labels.view(np.int32) if labels.dtype == np.uint32 \
                     else labels.astype(np.int32, copy=False)
        nmax = int(labels.max()) if labels.size else 0

        if despur and nmax > 0:
            # Label-aware despur pre-pass in C++ (delete_spurs_labels).
            # Wipes spurs AND 1-voxel-thick straight interior pixels
            # (including N-D diagonals) in a single mark-and-apply
            # over the whole image, catching 1-px inter-cell bridges.
            # Falls through to the cc_label_per_label branch below,
            # which handles disjoint-component splitting and area
            # filtering with the same numpy-vectorized logic.
            labels_i32_pre, _n_pre = _b.delete_spurs_labels(
                np.ascontiguousarray(labels_i32),
                threshold=1, max_iters=20, remove_thin=True)
            labels_i32 = labels_i32_pre
            labels = labels_i32.view(np.uint32) \
                if labels_i32.dtype == np.int32 \
                else labels_i32.astype(np.uint32, copy=False)
            nmax = int(labels.max()) if labels.size else 0

        if nmax > 0:
            # One label-aware CCL pass yields every component of every
            # input label, plus the source label of each component.
            comp_labels, n_total, source_per_comp = _b.cc_label_per_label(
                np.ascontiguousarray(labels_i32), conn=ndim,
            )
            if n_total > 0:
                comp_areas = _regionprops(comp_labels, n_total)['area']

                # Group components by source label: stable-sort packs
                # same-source components into contiguous slices that
                # searchsorted can then index by source value.
                sort_idx = np.argsort(source_per_comp, kind='stable')
                sorted_sources = source_per_comp[sort_idx]
                unique_sources = np.unique(source_per_comp)
                unique_sources = unique_sources[unique_sources > 0]

                # remap[c+1] is the new label value for component (c+1).
                # remap[0] = 0 keeps background pixels as background.
                remap = np.zeros(n_total + 1, dtype=np.int32)
                cur_max = nmax
                for j in unique_sources:
                    lo = int(np.searchsorted(sorted_sources, j, side='left'))
                    hi = int(np.searchsorted(sorted_sources, j, side='right'))
                    comp_indices = sort_idx[lo:hi]
                    if comp_indices.size == 0:
                        continue
                    areas_for_j = comp_areas[comp_indices]
                    order = np.argsort(-areas_for_j, kind='stable')
                    if comp_indices.size > 1 and verbose:
                        print('Warning - found mask with disjoint label.')
                    # Largest component keeps source label j; smaller disjoint
                    # parts get fresh labels (cur_max+1) or are dropped if too
                    # small. Threshold is uniform across ranks: components with
                    # area strictly less than min_area are dropped (i.e.
                    # min_area=9 keeps 9-pixel components, drops 8-pixel ones).
                    for rank, k in enumerate(order):
                        ci = int(comp_indices[k])
                        area = int(areas_for_j[k])
                        if area < min_area:
                            if verbose:
                                if rank == 0:
                                    print('Warning - found mask area less than', min_area)
                                    print('Removing it.')
                                else:
                                    print('secondary disjoint part smaller than min_area. Removing it.')
                            continue
                        if rank == 0:
                            remap[ci + 1] = int(j)
                        else:
                            if verbose:
                                print('secondary disjoint part bigger than min_area, relabeling. Area:', area,
                                      'Label value:', int(j))
                            cur_max += 1
                            remap[ci + 1] = cur_max

                labels = remap[comp_labels].astype(np.uint32, copy=False)

    # Compact to 1..N and downcast to the smallest unsigned int that fits.
    with _use(_engine) as engine:
        out, n_used = engine._expand.format_labels(
            np.ascontiguousarray(labels.astype(np.int32)),
            first_seen=True,
        )
    if n_used <= 0xFF:
        return out.astype(np.uint8, copy=False)
    if n_used <= 0xFFFF:
        return out.astype(np.uint16, copy=False)
    return out.astype(np.uint32, copy=False)


def delete_spurs(arr, hole_threshold=5, *, mode="cardinal",
                 threshold=None, max_iter=-1, kind="auto",
                 max_iters=None, remove_thin=False):
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
    pruning. Pass 0 to skip hole filling entirely.

    ``mode`` (binary mode) — connectivity used by the endpoint check:

    * ``"cardinal"`` (default) — face neighbors only (2·ndim of them).
      Catches pixels sticking out of a flat boundary; matches the
      omnipose external-spur rule. Aggressive; fewer iterations to
      converge.
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
    default 20 caps at 20 rounds.

    ``remove_thin`` (label mode only, default ``False``) — also zero
    1-voxel-thick straight interior pixels in the same pass. A pixel
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
        arr_u8 = arr.astype(np.uint8, copy=False)
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
    return _b.delete_spurs_labels(arr32, threshold=thr, max_iters=rounds,
                                    remove_thin=bool(remove_thin))