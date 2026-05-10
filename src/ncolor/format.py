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
        labels_i32 = labels.view(np.int32) if labels.dtype == np.uint32 \
                     else labels.astype(np.int32, copy=False)
        nmax = int(labels.max()) if labels.size else 0

        if despur:
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


def delete_spurs(mask, hole_threshold=5):
    """N-D skeleton cleanup: fill bg holes ≤ ``hole_threshold`` pixels
    (face-connected), then iteratively strip endpoints until no pixel
    has exactly one foreground neighbour. Endpoint connectivity is
    8-conn for 2D, face-only for ndim ≥ 3.
    """
    from ._backend import _impl as _b
    arr = np.ascontiguousarray(mask).astype(np.uint8, copy=False)
    return _b.delete_spurs(arr, int(hole_threshold))