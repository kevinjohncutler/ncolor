import multiprocessing as mp

import numpy as np
import pytest

import ncolor


def _make_spheres_3d(shape=(20, 20, 20), n_spheres=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    labels = np.zeros(shape, dtype=np.int32)
    max_attempts = n_spheres * 20
    attempts = 0
    label_id = 1
    radius_max = max(2, min(shape) // 6)
    while label_id <= n_spheres and attempts < max_attempts:
        attempts += 1
        radius = rng.integers(2, radius_max + 1)
        cz = rng.integers(radius, shape[0] - radius)
        cy = rng.integers(radius, shape[1] - radius)
        cx = rng.integers(radius, shape[2] - radius)
        mask = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        if np.any(labels[mask] != 0):
            continue
        labels[mask] = label_id
        label_id += 1
    return labels


def test_ncolor_3d_spheres_no_conflicts():
    labels = _make_spheres_3d()
    colored, conflicts = ncolor.label(labels, expand=False, return_conflicts=True)
    assert colored.shape == labels.shape
    assert colored.dtype.kind in "iu"
    assert conflicts == 0


def test_ncolor_3d_meshgrid_no_conflicts():
    size = 12
    labels = np.arange(1, size ** 3 + 1, dtype=np.int32).reshape((size, size, size))
    colored, conflicts = ncolor.label(labels, expand=False, return_conflicts=True)
    assert colored.shape == labels.shape
    assert colored.dtype.kind in "iu"
    assert conflicts == 0


def _make_dense_seeds_3d(shape, n_seeds, seed):
    """Single-voxel Poisson seeds. After ``expand=True`` these tessellate
    into a dense 3D Voronoi whose adjacency graph has high average degree
    (~12), so the number of distinct edges exceeds the old find_pairs
    hashtable size of ``2 * n_fwd * max_label`` (= 6 * n_cells at conn=1).
    """
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, dtype=np.int32)
    lo = np.array([2, 2, 2])
    hi = np.array(shape) - 2
    pts = rng.integers(lo, hi, size=(n_seeds, 3))
    for i, (z, y, x) in enumerate(pts, 1):
        m[z, y, x] = i
    return m


def _label_worker(labels, q):
    _, conflicts = ncolor.label(labels, expand=True, return_conflicts=True)
    q.put(int(conflicts))


def test_3d_dense_voronoi_does_not_hang_findpairs():
    """Regression: a dense 3D Voronoi where the distinct-edge count exceeds
    the find_pairs hashtable's initial size used to fill the open-addressing
    table to 100%, sending the linear-probe insert into an infinite loop
    (ncolor.label never returned). The bounded probe + retry-on-full sizing
    must now complete with a valid coloring.

    Config (100**3, 341 single-voxel seeds, seed=3) produces ~2090 distinct
    6-connected edges vs the old table size of ipow2(6*341) = 2048 — i.e. a
    guaranteed overflow on the pre-fix code path.

    Run in a subprocess with a hard timeout so a re-introduced hang fails
    the test instead of stalling the whole suite forever.
    """
    labels = _make_dense_seeds_3d((100, 100, 100), n_seeds=341, seed=3)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_label_worker, args=(labels, q))
    proc.start()
    proc.join(timeout=60)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        pytest.fail("ncolor.label(expand=True) hung on dense 3D Voronoi "
                    "(find_pairs hashtable overflow regression)")
    assert not q.empty(), "worker died without returning a result"
    assert q.get() == 0   # valid coloring, no adjacent same-color cells
