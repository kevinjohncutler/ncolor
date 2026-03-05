"""
Euclidean Voronoi expand_labels — parallel separable parabolic-envelope algorithm.

Reference: Felzenszwalb & Huttenlocher (2012). Works for any ndim.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _expand_batch_transpose(src_a, src_b, dst_a, dst_b):
    """Blocked parallel batched transpose: (A,B,C) → (A,C,B) for two arrays.
    Parallelises over all (a, b-tile, c-tile) combinations so both small-B and
    small-C cases distribute work across threads."""
    A, B, C = src_a.shape
    Bi = 64
    n_b = (B + Bi - 1) // Bi
    n_c = (C + Bi - 1) // Bi
    for i in prange(A * n_b * n_c):
        a  = i // (n_b * n_c)
        bc = i %  (n_b * n_c)
        b0 = (bc // n_c) * Bi
        c0 = (bc %  n_c) * Bi
        for b in range(b0, min(b0 + Bi, B)):
            for c in range(c0, min(c0 + Bi, C)):
                dst_a[a, c, b] = src_a[a, b, c]
                dst_b[a, c, b] = src_b[a, b, c]


@njit(cache=True, parallel=True)
def _expand_envelope_pass(h_lbl, h_dist):
    """
    Parabolic lower-envelope pass for one axis (Felzenszwalb & Huttenlocher 2012).
    h_lbl:  (n_slices, N) int32 — 0 = no seed; updated in-place to nearest label.
    h_dist: (n_slices, N) int32 — accumulated squared distance; updated in-place.
    """
    n_slices, N = h_lbl.shape
    for s in prange(n_slices):
        v   = np.empty(N, dtype=np.int32)
        lbl = np.empty(N, dtype=np.int32)
        g   = np.empty(N, dtype=np.int32)
        z   = np.empty(N + 1, dtype=np.float64)
        k   = np.int32(0)
        for i in range(N):
            if h_lbl[s, i] == 0:
                continue
            gi  = np.int32(h_dist[s, i])
            fi  = np.float64(i)
            gf  = np.float64(gi)
            new_z = np.float64(-1e18)
            while k > 0:
                top = k - 1
                ft  = np.float64(v[top])
                sv  = (gf + fi*fi - np.float64(g[top]) - ft*ft) / (2.0 * (fi - ft))
                if sv > z[top]:
                    new_z = sv
                    break
                k -= np.int32(1)
            z[k] = new_z; v[k] = np.int32(i); lbl[k] = h_lbl[s, i]; g[k] = gi
            k += np.int32(1)
        if k == 0:
            continue
        j = np.int32(0)
        for i in range(N):
            while j + 1 < k and np.float64(i) >= z[j + 1]:
                j += np.int32(1)
            di = np.int32(i) - v[j]
            h_lbl[s, i]  = lbl[j]
            h_dist[s, i] = g[j] + di * di


_expand_buf_shape = (0,)
_expand_buf_dtype = None
_expand_h_lbl   = _expand_h_dist = _expand_h_lbl_T = _expand_h_dist_T = _expand_out = None

def _ensure_expand_bufs(shape, dtype):
    global _expand_buf_shape, _expand_buf_dtype
    global _expand_h_lbl, _expand_h_dist, _expand_h_lbl_T, _expand_h_dist_T, _expand_out
    if shape == _expand_buf_shape and dtype == _expand_buf_dtype:
        return
    _expand_h_lbl  = np.empty(shape, dtype=np.int32)
    _expand_h_dist = np.empty(shape, dtype=np.int32)
    _expand_out    = np.empty(shape, dtype=dtype)
    if len(shape) >= 2:
        _expand_h_lbl_T  = np.empty(shape, dtype=np.int32)
        _expand_h_dist_T = np.empty(shape, dtype=np.int32)
    _expand_buf_shape = shape
    _expand_buf_dtype = dtype


def expand_labels(label_image):
    """
    Expand labels into every background pixel using nearest-label Euclidean
    Voronoi assignment.

    Uses a parallel separable parabolic-envelope algorithm (Felzenszwalb &
    Huttenlocher 2012) implemented in Numba. Works for any ndim.
    """
    _ensure_expand_bufs(label_image.shape, label_image.dtype)
    ndim = label_image.ndim
    np.copyto(_expand_h_lbl, label_image, casting='unsafe')
    _expand_h_dist[:] = 0

    for ax in range(ndim - 1, -1, -1):
        n = label_image.shape[ax]
        if ax == ndim - 1:
            _expand_envelope_pass(_expand_h_lbl.reshape(-1, n),
                                  _expand_h_dist.reshape(-1, n))
        else:
            # All non-last axes: reshape to (A, n, C) and use blocked batched transpose.
            # A = axes before ax, C = axes after ax; no numpy copies needed.
            A = int(np.prod(label_image.shape[:ax])) if ax > 0 else 1
            C = int(np.prod(label_image.shape[ax + 1:]))
            hl = _expand_h_lbl.reshape(A, n, C)
            hd = _expand_h_dist.reshape(A, n, C)
            tl = _expand_h_lbl_T.reshape(A, C, n)
            td = _expand_h_dist_T.reshape(A, C, n)
            _expand_batch_transpose(hl, hd, tl, td)
            _expand_envelope_pass(tl.reshape(-1, n), td.reshape(-1, n))
            _expand_batch_transpose(tl, td, hl, hd)

    np.copyto(_expand_out, _expand_h_lbl, casting='unsafe')
    return _expand_out
