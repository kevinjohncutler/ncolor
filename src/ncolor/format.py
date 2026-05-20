import numpy as np


# Persistent thread pool; constructing per call costs ~5-10 ms.
_FORMAT_ENGINE = None


def _get_format_engine():
    global _FORMAT_ENGINE
    if _FORMAT_ENGINE is None:
        from ._backend import ExpandEngine
        _FORMAT_ENGINE = ExpandEngine(-1)
    return _FORMAT_ENGINE


def format_labels(labels, clean=False, min_area=9, despur=False,
                  verbose=False, background=None, ignore=False,
                  first_seen=False):
    """Compact labels into background=0, cells 1..N.

    ``clean=True`` splits disjoint components per label and drops
    components below ``min_area``. ``despur=True`` additionally runs
    :func:`delete_spurs` on each label's mask before splitting.

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
        eng = _get_format_engine()
        arr = np.ascontiguousarray(labels)
        out, _n = eng.format_labels(arr, first_seen=bool(first_seen))
        return out

    # Cellpose stores labels inside float arrays; cast back to int.
    # Some segmenters use -1 as background, so use a signed dtype here.
    labels = labels.copy().astype('int32')
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
        from .color import connected_components, regionprops as _regionprops
        from ._backend import _impl as _b

        ndim = labels.ndim
        # uint32 → int32 by view (zero-copy, same byte width). Safe
        # because no realistic label image has > 2^31 distinct values;
        # the upper bit is always clear so reinterpretation is identical.
        labels_i32 = labels.view(np.int32) if labels.dtype == np.uint32 \
                     else labels.astype(np.int32, copy=False)
        nmax = int(labels.max()) if labels.size else 0

        if despur:
            # Fast label-aware pre-pass: wipe spurs AND 1-voxel-thick
            # straight interior pixels (axis-aligned + 2D diagonals)
            # in a single mark-and-apply over the whole image. Catches
            # 1-px inter-cell bridges that the per-label binary despur
            # below would otherwise eat one ring at a time. The result
            # is bit-equivalent to running the per-label loop to
            # convergence, just produced in one pass.
            if nmax > 0:
                labels_i32_pre, _n_pre = _b.delete_spurs_labels(
                    np.ascontiguousarray(labels_i32),
                    threshold=1, max_iters=20, remove_thin=True)
                labels_i32 = labels_i32_pre
                labels = labels_i32.view(np.uint32) \
                    if labels_i32.dtype == np.int32 \
                    else labels_i32.astype(np.uint32, copy=False)
                nmax = int(labels.max()) if labels.size else 0
            # delete_spurs mutates the per-label mask between rounds, so
            # the despur path stays per-label (bbox-cropped).
            global_props = _regionprops(np.ascontiguousarray(labels_i32), nmax) if nmax > 0 else None
            bbox_min_all = global_props['bbox_min'] if global_props else None
            bbox_max_all = global_props['bbox_max'] if global_props else None
            areas_all    = global_props['area']     if global_props else None
            cur_max = nmax
            for j in range(1, nmax + 1):
                if areas_all[j - 1] == 0:
                    continue
                bb_lo = bbox_min_all[j - 1]; bb_hi = bbox_max_all[j - 1]
                slc = tuple(slice(int(bb_lo[d]), int(bb_hi[d])) for d in range(ndim))
                crop_view = labels[slc]
                mask = (crop_view == j)
                crop_view[mask] = 0
                mask = delete_spurs(mask)
                crop_view[mask] = j

                lbl, n_comp = connected_components(mask.astype(np.uint8), conn=ndim)
                if n_comp == 0:
                    continue
                sub = _regionprops(lbl, n_comp)
                sub_areas = sub['area']; sub_bb_lo = sub['bbox_min']; sub_bb_hi = sub['bbox_max']
                order = np.argsort(-sub_areas, kind='stable')
                if n_comp > 1 and verbose:
                    print('Warning - found mask with disjoint label.')
                for rank, comp_idx in enumerate(order):
                    comp_id = int(comp_idx) + 1
                    area = int(sub_areas[comp_idx])
                    cb_lo = sub_bb_lo[comp_idx]; cb_hi = sub_bb_hi[comp_idx]
                    sub_slc = tuple(slice(int(cb_lo[d]), int(cb_hi[d])) for d in range(ndim))
                    local = np.argwhere(lbl[sub_slc] == comp_id)
                    offset = bb_lo + cb_lo
                    global_t = tuple((local + offset).T)
                    if rank == 0:
                        if area <= min_area:
                            labels[global_t] = background
                            if verbose:
                                print('Warning - found mask area less than', min_area)
                                print('Removing it.')
                    else:
                        if area < min_area:
                            labels[global_t] = background
                        else:
                            cur_max += 1
                            labels[global_t] = cur_max
        elif nmax > 0:
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
                    # small. Note the asymmetric threshold: rank==0 uses
                    # area<=min_area, rank>0 uses area<min_area — matches the
                    # original skimage+fastremap pipeline at exactly min_area.
                    for rank, k in enumerate(order):
                        ci = int(comp_indices[k])
                        area = int(areas_for_j[k])
                        if rank == 0:
                            if area <= min_area:
                                if verbose:
                                    print('Warning - found mask area less than', min_area)
                                    print('Removing it.')
                            else:
                                remap[ci + 1] = int(j)
                        else:
                            if area < min_area:
                                if verbose:
                                    print('secondary disjoint part smaller than min_area. Removing it.')
                            else:
                                if verbose:
                                    print('secondary disjoint part bigger than min_area, relabeling. Area:', area,
                                          'Label value:', int(j))
                                cur_max += 1
                                remap[ci + 1] = cur_max

                labels = remap[comp_labels].astype(np.uint32, copy=False)

    # Compact to 1..N and downcast to the smallest unsigned int that fits.
    eng = _get_format_engine()
    out, n_used = eng.format_labels(
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
      1-voxel-wide skeletons in 3D (their interiors have face=2 but
      total=2, both equal to ndim=3 minus 1 — under ``threshold=ndim``
      they survive cardinal but fall below total). Use this when the
      input may contain genuine thin features you want to keep.

    ``threshold`` — binary mode default is ``None`` → ``ndim`` (pixel
    is a spur if fg-neighbor count is in ``[1, threshold)``); label
    mode default is 1.

    ``max_iter`` (binary) / ``max_iters`` (label) — cap the iterative
    pruning loop. Binary default ``-1`` runs to convergence; label
    default 20 caps at 20 rounds.

    ``remove_thin`` (label mode only, default ``False``) — also zero
    1-voxel-thick straight interior pixels in the same pass. A pixel
    qualifies iff it has exactly two same-label 8-connectivity
    neighbors that sit at opposite offsets (axis-aligned in any ndim,
    plus 2D diagonals NW-SE / NE-SW). Useful when expand_labels
    leaves 1-px bridges between cells that would otherwise need many
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
    # kind == "labels"
    arr32 = arr.astype(np.int32, copy=False)
    thr = int(threshold) if threshold is not None else 1
    rounds = int(max_iters) if max_iters is not None else (
        20 if max_iter == -1 else int(max_iter))
    return _b.delete_spurs_labels(arr32, threshold=thr, max_iters=rounds,
                                    remove_thin=bool(remove_thin))