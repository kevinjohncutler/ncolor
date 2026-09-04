"""Build the benchmark corpus once and save it, so both builds under
comparison see byte-identical inputs.

Generating the arrays inside each build would let a change to the
generators masquerade as a performance difference, and would put the
random draws in a different order for each. One .npz, loaded by both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def boxes(shape, n, seed, rmin, rmax):
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, np.int32)
    lo = min(shape)
    centers = rng.integers(rmax, max(rmax + 1, lo - rmax), size=(n, len(shape)))
    radii = rng.integers(rmin, rmax, size=n)
    for i, (c, r) in enumerate(zip(centers, radii), 1):
        sl = tuple(slice(max(0, x - r), x + r) for x in c)
        m[sl] = i
    return m


def disks(H, n, seed):
    """Round cells: the shape real segmentations actually have."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, H), np.int32)
    yy, xx = np.ogrid[:H, :H]
    for i, (cy, cx, r) in enumerate(
            zip(rng.integers(0, H, n), rng.integers(0, H, n),
                rng.integers(max(3, H // 120), max(6, H // 40), n)), 1):
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def voronoi(H, n, seed):
    """Every region touches its neighbors: the densest adjacency graph a
    real image produces, and no background at all."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, H, size=(n, 2))
    yy, xx = np.mgrid[:H, :H]
    d = ((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2)
    return (np.argmin(d, axis=-1) + 1).astype(np.int32)


def filaments(H, n, seed):
    """Long thin objects, as bacteria or neurites: high perimeter per
    area, so many adjacencies per cell."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, H), np.int32)
    for i in range(1, n + 1):
        y, x = rng.integers(0, H, 2)
        L, w = rng.integers(H // 10, H // 4), rng.integers(1, 4)
        if rng.random() < 0.5:
            m[y:y + w, x:x + L] = i
        else:
            m[y:y + L, x:x + w] = i
    return m


def tiles(H, block):
    """As many cells per pixel as the scan will ever see, but a graph
    ncolor can actually color.

    A random label per pixel is the obvious stress case and the wrong
    one: 65536 mutually adjacent cells need thousands of colors, the
    uint8 output saturates at 255, and the cells that lose out come back
    as background. Which ones depends on how far the picker gets inside
    its wall-clock budget, so the output moves with machine load. A grid
    of small blocks keeps the punishing cell count and the huge adjacency
    graph while staying 4-colorable and deterministic.
    """
    n = H // block
    ids = np.arange(n * n, dtype=np.int32).reshape(n, n) + 1
    return np.kron(ids, np.ones((block, block), np.int32))


def sparse(H, n, seed, frac=0.02):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, H), np.int32)
    r = max(2, int(np.sqrt(frac * H * H / max(1, n) / np.pi)))
    for i, (cy, cx) in enumerate(zip(rng.integers(0, H, n), rng.integers(0, H, n)), 1):
        m[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = i
    return m


def wide_ids(H, n, seed):
    """Labels that are not 1..N: format_labels has to compact them."""
    rng = np.random.default_rng(seed)
    base = boxes((H, H), n, seed, 4, max(6, H // 40))
    ids = rng.choice(np.arange(1, 2 ** 20), size=int(base.max()) + 1, replace=False)
    ids[0] = 0
    return ids[base].astype(np.int32)


def build():
    c = {}
    for s in (0, 1, 2):                       # several draws of the same kind
        c[f"boxes2d_1024_s{s}"] = boxes((1024, 1024), 130, s, 8, 34)
    c["boxes2d_512"] = boxes((512, 512), 40, 0, 6, 20)
    c["boxes2d_2048"] = boxes((2048, 2048), 520, 0, 12, 68)
    c["boxes2d_4096"] = boxes((4096, 4096), 2100, 0, 16, 136)
    c["aniso_512x4096"] = boxes((512, 4096), 260, 0, 8, 34)
    c["disks_1024"] = disks(1024, 600, 0)
    c["disks_2048"] = disks(2048, 2400, 1)
    c["voronoi_512"] = voronoi(512, 400, 0)
    c["voronoi_1024"] = voronoi(1024, 1200, 1)
    c["filaments_1024"] = filaments(1024, 700, 0)
    c["tiles_1024_b4"] = tiles(1024, 4)          # 65536 cells, 4-colorable
    c["sparse_2048"] = sparse(2048, 900, 0)
    c["many_labels_1024"] = disks(1024, 9000, 2)
    c["wide_ids_1024"] = wide_ids(1024, 500, 0)
    c["boxes3d_64"] = boxes((64,) * 3, 120, 0, 3, 8)
    c["boxes3d_128"] = boxes((128,) * 3, 700, 0, 3, 10)
    c["boxes3d_192"] = boxes((192,) * 3, 1800, 1, 3, 12)
    fx = REPO / "test_files" / "synthetic_800.npz"
    if fx.exists():
        c["fixture_800"] = np.load(fx)["labels"].astype(np.int32)
    return c


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "bench_outputs" / "corpus.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = build()
    np.savez_compressed(out, **cases)
    total = sum(a.nbytes for a in cases.values())
    print(f"wrote {out}  {len(cases)} cases, {total / 1e6:.0f} MB in memory")
    for k, v in cases.items():
        print(f"  {k:22s} {str(v.shape):18s} labels={int(v.max()):7d} "
              f"fill={float((v > 0).mean()):.2f}")
