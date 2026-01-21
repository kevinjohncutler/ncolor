import numpy as np
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
