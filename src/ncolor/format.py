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
        # Lazy in-function import: ncolor.color top-imports format.py, so
        # a top-level import here would deadlock module init.
        from .color import connected_components, regionprops as _regionprops
        inds = np.unique(labels)
        for j in inds[inds > background]:
            mask = (labels == j)
            if despur:
                labels[mask] = 0
                mask = delete_spurs(mask)
                labels[mask] = j

            lbl, n_comp = connected_components(mask.astype(np.uint8), conn=mask.ndim)
            if n_comp == 0:
                continue
            areas = _regionprops(lbl, n_comp)['area']
            # Stable descending-area order matches skimage's
            # regionprops(...).sort(key=lambda r: r.area, reverse=True):
            # ties keep ascending-component-id order. ncolor.cc_label
            # numbers components in raster scan order, identical to
            # skimage.measure.label, so the per-component decisions land
            # on the same coords.
            order = np.argsort(-areas, kind='stable')

            if n_comp > 1 and verbose:
                print('Warning - found mask with disjoint label.')

            cur_max = int(labels.max())
            for rank, comp_idx in enumerate(order):
                comp_id = int(comp_idx) + 1  # cpp cc_label uses 1..N
                area = int(areas[comp_idx])
                coords_t = tuple(np.argwhere(lbl == comp_id).T)
                if rank == 0:
                    if area <= min_area:
                        labels[coords_t] = background
                        if verbose:
                            print('Warning - found mask area less than', min_area)
                            print('Removing it.')
                else:
                    if area < min_area:
                        labels[coords_t] = background
                        if verbose:
                            print('secondary disjoint part smaller than min_area. Removing it.')
                    else:
                        if verbose:
                            print('secondary disjoint part bigger than min_area, relabeling. Area:', area,
                                  'Label value:', np.unique(labels[coords_t]))
                        cur_max += 1
                        labels[coords_t] = cur_max

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