"""Compare the optimized clean kernels to scalar sweeps and exact degrees.

The reference deliberately has no SIMD, transposes, saturated neighbor
counts or lazy recounts. Its exact-degree queue catches double decrements
when several initially removed pixels border a lazily recounted pixel.
"""
from collections import deque
from itertools import product

import numpy as np
import pytest

import ncolor


def _clean_reference(image, p, wrap):
    shape = image.shape
    labels = image.copy()
    dist = np.where(labels, 0, 2**29).astype(np.int64)
    barrier = np.zeros(shape, bool)
    for ax in reversed(range(image.ndim)):
        for fixed in np.ndindex(shape[:ax] + shape[ax + 1:]):
            sel = fixed[:ax] + (slice(None),) + fixed[ax:]
            values, costs = labels[sel].copy(), dist[sel].copy()
            blocked = barrier[sel]
            width = len(values)
            if p == 2:
                positions = range(-width, 2 * width) if wrap else range(width)
                seeds = [(i, int(values[i % width]), int(costs[i % width]))
                         for i in positions if values[i % width]]
                for x in range(width):
                    if blocked[x] or not seeds:
                        continue
                    # At equal L2 distance the higher seed coordinate wins.
                    i, value, cost = min(seeds, key=lambda t: (t[2] + (x - t[0])**2, -t[0]))
                    labels[sel][x] = value
                    dist[sel][x] = cost + (x - i)**2
            else:
                # L1 uses strict-improvement relaxations; keep its documented
                # sweep-order tie break rather than imposing L2's tie rule.
                def relax(dst, src):
                    if (not blocked[dst] and not blocked[src]
                            and costs[src] + 1 < costs[dst]):
                        costs[dst] = costs[src] + 1
                        values[dst] = values[src]

                for x in range(1, width):
                    relax(x, x - 1)
                for x in reversed(range(width - 1)):
                    relax(x, x + 1)
                if wrap and width > 1:
                    relax(0, width - 1)
                    for x in range(1, width):
                        relax(x, x - 1)
                    relax(width - 1, 0)
                    for x in reversed(range(width - 1)):
                        relax(x, x + 1)
                labels[sel], dist[sel] = values, costs

        if image.ndim - ax < 2:
            continue
        offsets = [(0,) * ax + o
                   for o in product((-1, 0, 1), repeat=image.ndim - ax) if any(o)]
        faces = [o for o in offsets if np.count_nonzero(o) == 1]

        def neighbor(pos, offset):
            q = tuple(x + v for x, v in zip(pos, offset))
            if wrap:
                return tuple(x % n for x, n in zip(q, shape))
            if all(0 <= x < n for x, n in zip(q, shape)):
                return q
            return None

        degree = np.zeros(shape, np.int32)
        initial = []
        for pos in np.ndindex(shape):
            value = labels[pos]
            if not value:
                continue
            same = []
            for offset in offsets:
                q = neighbor(pos, offset)
                if q is not None and labels[q] == value:
                    same.append(offset)
            degree[pos] = sum(np.count_nonzero(o) == 1 for o in same)
            antipodal = len(same) == 2 and all(x == -y for x, y in zip(*same))
            if degree[pos] <= 1 or antipodal:
                initial.append((pos, int(value)))

        # Exact degrees allow simultaneous initial removals; each removed
        # edge is subtracted exactly once as its source leaves the queue.
        queue = deque(initial)
        for pos, _ in initial:
            labels[pos], barrier[pos] = 0, True
        while queue:
            pos, value = queue.popleft()
            for offset in faces:
                q = neighbor(pos, offset)
                if q is not None and labels[q] == value:
                    degree[q] -= 1
                    if degree[q] <= 1:
                        queue.append((q, value))
                        labels[q], barrier[q] = 0, True
    return labels


@pytest.mark.parametrize('ndim', [2, 3, 4])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('wrap', [False, True])
def test_clean_matches_scalar_reference(ndim, seed, p, wrap):
    rng = np.random.default_rng(seed)
    shape = (5,) * ndim
    labels = np.where(rng.random(shape) < .1,
                      rng.integers(1, 5, size=shape), 0).astype(np.int32)
    expected = _clean_reference(labels, p, wrap)
    actual = ncolor.expand_labels(labels, p=p, mode='clean', wrap=wrap)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('wrap', [False, True])
def test_clean_parallel_matches_serial_with_barriers(p, wrap):
    rng = np.random.default_rng(105)
    shape = (6, 7, 8, 9)
    labels = np.where(rng.random(shape) < .06,
                      rng.integers(1, 12, size=shape), 0).astype(np.int32)
    serial = ncolor.Engine(n_threads=1).expand_labels(labels, p=p, mode='clean', wrap=wrap)
    parallel_engine = ncolor.Engine(n_threads=4)
    for _ in range(3):
        parallel = parallel_engine.expand_labels(labels, p=p, mode='clean', wrap=wrap)
        np.testing.assert_array_equal(parallel, serial)
