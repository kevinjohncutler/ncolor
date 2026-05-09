import numpy as np


# Module-level ExpandEngine singleton. Constructing one per call adds
# ~5-10 ms of pool-spinup overhead — matches the rationale in color.py's
# _SOLVER. Engine is thread-safe for sequential format_labels calls.
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
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks below min_area.
    min_area default is 9px.
    Optional ignore flag: 0 is now 'ignore' and 1 is background. We do not want to shift 1->0 in that case.

    first_seen (default False): when False, the new label assigned to a
    source value is its rank among present values (ascending-source,
    parallel build, fast). When True, labels are assigned in input scan
    order — matches ``fastremap.renumber`` bit-for-bit, serial build,
    ~2× slower. Set to True only if downstream code depends on the
    legacy fastremap ordering.
    """
    # Hot path: simple compaction (no cleanup, default min-shift) ->
    # cpp engine. Skips the numpy.copy + astype + np.min + fastremap
    # round-trip that this function used to do per call.
    if (not clean and not ignore and background is None and not verbose):
        try:
            eng = _get_format_engine()
        except ImportError:
            pass  # fall through to legacy path
        else:
            # Pass the original-dtype array straight to cpp. ExpandEngine
            # casts to int32 in parallel inside the released-GIL block via
            # cast_to_int32 — no numpy.astype + .copy() round-trip outside
            # the GIL release. Output uses ascending-source numbering: the
            # new label assigned to source value k is its rank among
            # present values. Differs from fastremap.renumber's
            # input-order numbering by a permutation, but both produce a
            # valid 1..N compaction; downstream callers treat label
            # values as opaque identifiers.
            arr = np.ascontiguousarray(labels)
            out, _n = eng.format_labels(arr, first_seen=bool(first_seen))
            return out

    # Legacy path: clean=True / ignore / custom background / verbose.
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32.
    labels = labels.copy()
    labels = labels.astype('int32') #uint vs int
    if background is None:
        # Treat min as bg ONLY when negative (e.g. -1 used as bg by some
        # segmenters). For min >= 0 the input either already has bg at 0
        # or has no bg (every pixel labeled — typical for already-expanded
        # label maps); shifting in those cases would absorb the smallest
        # cell into the bg. Mirrors the cpp fix in format_labels_inplace.
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

    # optional cleanup
    if clean:
        # The despur=True branch still runs per-label (it mutates the
        # mask between cc passes); the common despur=False path uses a
        # single global cpp call into cc_label_per_label which returns
        # (component_labels, n_components, source_label_per_component)
        # in one round-trip. That avoids paying ~290 µs of Python⇄cpp
        # boundary overhead per input label, which was the bottleneck
        # after bbox-cropping was already in place.
        from .color import connected_components, regionprops as _regionprops
        from ._backend import _impl as _b

        ndim = labels.ndim
        labels_i32 = labels.view(np.int32) if labels.dtype == np.uint32 \
                     else labels.astype(np.int32, copy=False)
        nmax = int(labels.max()) if labels.size else 0

        if despur:
            # Per-label loop required: delete_spurs needs the binary
            # mask of just one label, and despur mutates the labels
            # buffer in-place between iterations. Bbox-cropped path,
            # same as before — only used when the user explicitly
            # passes despur=True.
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
            # Fast path: one global label-aware CCL gives every
            # component of every input label in a single pass, plus
            # a (n_components,) array of source labels.
            comp_labels, n_total, source_per_comp = _b.cc_label_per_label(
                np.ascontiguousarray(labels_i32), conn=ndim,
            )
            if n_total > 0:
                # One regionprops call: areas for all components.
                comp_props = _regionprops(comp_labels, n_total)
                comp_areas = comp_props['area']

                # Group components by source label via a stable sort on
                # source_per_comp. After sorting, components sharing a
                # source label are contiguous, located by searchsorted.
                sort_idx = np.argsort(source_per_comp, kind='stable')
                sorted_sources = source_per_comp[sort_idx]
                unique_sources = np.unique(source_per_comp)
                unique_sources = unique_sources[unique_sources > 0]

                # remap[c+1] = the new label value for component c+1.
                # Bg pixels (comp_label == 0) read remap[0] = 0.
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
                        ci = int(comp_indices[k])  # 0-based
                        area = int(areas_for_j[k])
                        if rank == 0:
                            # Largest component: keep at j unless ≤ min_area.
                            if area <= min_area:
                                if verbose:
                                    print('Warning - found mask area less than', min_area)
                                    print('Removing it.')
                                # remap[ci + 1] stays 0 → drops to bg
                            else:
                                remap[ci + 1] = int(j)
                        else:
                            if area < min_area:
                                if verbose:
                                    print('secondary disjoint part smaller than min_area. Removing it.')
                                # remap[ci + 1] stays 0 → drops to bg
                            else:
                                if verbose:
                                    print('secondary disjoint part bigger than min_area, relabeling. Area:', area,
                                          'Label value:', int(j))
                                cur_max += 1
                                remap[ci + 1] = cur_max

                # Single fancy-index pass writes the cleaned labels.
                # Pixels where comp_labels == 0 (always-bg) read remap[0] = 0,
                # so this also clears any bg that was non-zero originally.
                labels = remap[comp_labels].astype(np.uint32, copy=False)

    # Final compaction — cpp ``format_labels`` with first_seen=True is
    # bit-identical to fastremap.renumber for raster-scan label
    # assignment. The cpp side always returns int32; downcast to the
    # smallest unsigned dtype that fits ``n_used`` to match
    # fastremap.refit's semantics on a uint32-typed input.
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
    """N-D skeleton cleanup. Pads input by 1, fills bg holes with
    pixel count ≤ ``hole_threshold`` (face-connected), then iteratively
    prunes endpoints (fg pixels with exactly one fg neighbour) until no
    pixels change in a pass. Returns a fresh boolean array of the input
    shape.

    Endpoint connectivity matches the original Python implementation:
    8-conn (full diagonal) for 2D, face-only for ndim ≥ 3.

    Implemented in cpp (``ncolor._backend._impl.delete_spurs``); no
    scikit-image / scipy required.
    """
    from ._backend import _impl as _b
    arr = np.ascontiguousarray(mask).astype(np.uint8, copy=False)
    return _b.delete_spurs(arr, int(hole_threshold))


# import sys
# sys.modules[__name__] = format_labels