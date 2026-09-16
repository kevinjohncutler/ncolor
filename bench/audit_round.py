"""Repeatable correctness, retained-memory, and performance audit.

Run with PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/audit_round.py
--seed 101 --output bench/audit_results/round_1.json. Requires psutil and
Shapely in addition to ncolor, unless --skip-geometry is explicit.
Seeds vary correctness cases only; all timed
and memory workloads stay identical across rounds. Run without other jobs
competing for CPU time. This supplements, rather than replaces, pytest.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import gc
from itertools import product
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil
try:
    import shapely
except ImportError:
    shapely = None

import ncolor
from ncolor import geo
from ncolor._backend import _impl


def pairs_of(edges):
    return {tuple(map(int, edge)) for edge in edges}


def contact_counts(image, conn, radius=1, wrap=False):
    """Exhaustive pixel-pair reference, independent of the native scan."""
    offsets = [d for d in product(range(-radius, radius + 1), repeat=image.ndim)
               if 0 < sum(x != 0 for x in d) <= conn
               and next((x for x in d if x), 0) > 0
               and all(size > 1 or x == 0 for size, x in zip(image.shape, d))]
    counts = Counter()
    for point in np.ndindex(image.shape):
        a = int(image[point])
        if not a:
            continue
        for delta in offsets:
            neighbor = tuple(x + d for x, d in zip(point, delta))
            if wrap:
                neighbor = tuple(x % size for x, size in zip(neighbor, image.shape))
            elif any(x < 0 or x >= size for x, size in zip(neighbor, image.shape)):
                continue
            b = int(image[neighbor])
            if b and b != a:
                counts[tuple(sorted((a, b)))] += 1
    return counts


def components_reference(image, conn):
    offsets = [d for d in product((-1, 0, 1), repeat=image.ndim)
               if 0 < sum(abs(x) for x in d) <= conn]
    result = np.zeros(image.shape, np.int32)
    sources = []
    for point in np.ndindex(image.shape):
        if not image[point] or result[point]:
            continue
        sources.append(image[point])
        result[point] = len(sources)
        queue = deque([point])
        while queue:
            current = queue.popleft()
            for delta in offsets:
                neighbor = tuple(x + d for x, d in zip(current, delta))
                if (all(0 <= x < size for x, size in zip(neighbor, image.shape))
                        and not result[neighbor] and image[neighbor] == image[point]):
                    result[neighbor] = len(sources)
                    queue.append(neighbor)
    return result, sources


def correctness(seed, include_geometry=True):
    rng = np.random.default_rng(seed)
    engine = ncolor.Engine(n_threads=4)
    expand = _impl.ExpandEngine(2)
    counts = Counter()
    for _ in range(150):
        shape = tuple(map(int, rng.integers(1, 7, size=int(rng.integers(1, 6)))))
        image = np.zeros(shape, np.int32)
        positions = rng.choice(image.size, min(image.size, 4), replace=False)
        image.flat[positions] = np.arange(1, len(positions) + 1)
        seeds = np.array(np.unravel_index(positions, shape)).T
        coords = np.indices(shape)
        for p in (1, 2):
            wrap = bool(rng.integers(2))
            costs = []
            for location in seeds:
                delta = abs(coords - location.reshape((-1,) + (1,) * len(shape)))
                if wrap:
                    delta = np.minimum(delta, np.array(shape).reshape(
                        (-1,) + (1,) * len(shape)) - delta)
                costs.append(np.sum(delta ** p, axis=0))
            costs = np.stack(costs)
            out, distance = expand.expand_labels_with_dist(image, p=p, wrap=wrap)
            selected = np.take_along_axis(costs, (out - 1)[None], axis=0)[0]
            np.testing.assert_array_equal(selected, costs.min(axis=0))
            np.testing.assert_allclose(distance, selected ** (1 / p))
            counts['expansion'] += 1

    for _ in range(60):
        shape = tuple(map(int, rng.integers(1, 5, size=int(rng.integers(1, 5)))))
        image = rng.integers(0, 6, shape, dtype=np.int32)
        conn = int(rng.integers(1, image.ndim + 1))
        expected, sources = components_reference(image, conn)
        actual, count, values = _impl.cc_label_per_label(image, conn=conn)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, sources)
        assert count == len(sources)
        expected, sources = components_reference(image != 0, conn)
        actual, count = ncolor.connected_components(image, conn=conn)
        np.testing.assert_array_equal(actual, expected)
        assert count == len(sources)
        props = ncolor.regionprops(image, n_labels=5)
        for value in range(1, 6):
            points = np.argwhere(image == value)
            assert props['area'][value - 1] == len(points)
            if len(points):
                np.testing.assert_array_equal(props['bbox_min'][value - 1], points.min(0))
                np.testing.assert_array_equal(props['bbox_max'][value - 1], points.max(0) + 1)
                np.testing.assert_allclose(props['centroid'][value - 1], points.mean(0))
        counts['components_and_properties'] += 1

    for dtype in (np.int16, np.int64, np.uint64, np.float64, np.bool_):
        for first_seen in (False, True):
            for background in (None, 0, 2):
                image = rng.integers(0, 6, (7, 8)).astype(dtype)
                if dtype == np.int64:
                    image -= 2
                    image *= 2 ** 35
                if dtype == np.uint64:
                    image *= 2 ** 35
                if dtype == np.float64:
                    image -= 2.4
                image = image[::-1, ::2]
                source = np.trunc(image) if dtype == np.float64 else image
                bg = min(0, source.min()) if background is None else background
                foreground = [x for x in source.flat if x != bg]
                order = list(dict.fromkeys(foreground)) if first_seen else sorted(set(foreground))
                mapping = {x: i + 1 for i, x in enumerate(order)}
                expected = np.array([mapping.get(x, 0) for x in source.flat]).reshape(source.shape)
                actual = engine.format_labels(image, background=background, first_seen=first_seen)
                np.testing.assert_array_equal(actual, expected)
                counts['formatting'] += 1

    for iteration in range(100):
        shape = (5,) * int(rng.integers(2, 4))
        image = rng.integers(0, 7, shape, dtype=np.int32)
        conn = int(rng.integers(1, image.ndim + 1))
        wrap = bool(rng.integers(2))
        signed = image * 15000 - 30000
        if iteration % 3 == 0:
            signed = -image
        expected = set(contact_counts(signed, conn, wrap=wrap))
        assert pairs_of(engine._solver.connect(signed, conn=conn, wrap=wrap)) == expected
        counts['signed_adjacency'] += 1
        image = engine.format_labels(image)
        p = int(rng.integers(1, 3))
        mode = ('standard', 'clean')[iteration % 2]
        expanded = iteration % 3 != 0
        radius = int(rng.integers(1, 3))
        min_contact = int(rng.integers(1, 5))
        working = engine.expand_labels(image, p=p, wrap=wrap, mode=mode) if expanded else image
        expected = {pair for pair, count in contact_counts(
            working, conn, radius, wrap).items() if count >= min_contact}
        options = dict(n=8, conn=conn, p=p, wrap=wrap, expand=expanded,
                       expand_mode=mode, connect_radius=radius, min_contact=min_contact,
                       soft_conn=0, soft_radius=0, max_depth=1,
                       weight_objective=(-1, 0, 1)[iteration % 3],
                       weight_mode=('min', 'max', 'mean', 'count', 'harmonic', 'mean_inv')[iteration % 6],
                       return_lut=True, return_n=True, return_conflicts=True)
        soft_conn = int(rng.integers(1, image.ndim + 1))
        soft_radius = int(rng.integers(1, 3))
        options.update(soft_conn=soft_conn, soft_radius=soft_radius)
        expected_soft = set(contact_counts(working, soft_conn, soft_radius, wrap)) - expected
        if iteration % 7 == 0:
            options['soft_extra_edges'] = np.empty((0, 2), np.int32)
            expected_soft = set()
        if iteration % 11 == 0:
            extra = (1, int(image.max()))
            options['extra_edges'] = np.array([extra], np.int32)
            expected.add(extra)
        lut, used, conflicts = engine.label(image, **options)
        assert pairs_of(engine._solver.get_last_soft_pairs()) == expected_soft
        assert conflicts == sum(lut[a] == lut[b] for a, b in expected) == 0
        assert used == len(np.unique(lut[1:]))
        if iteration % 10 == 0:
            engine.release_buffers()
        counts['raster_coloring'] += 1

    for iteration in range(90):
        shape = tuple(map(int, rng.integers(2, 7, size=int(rng.integers(2, 5)))))
        image = rng.integers(0, 7, shape, dtype=np.int32)
        options = dict(expand=iteration % 3 != 0, p=1 + iteration % 2,
                       expand_mode=('standard', 'clean')[iteration % 2],
                       wrap=bool(iteration % 2), conn=int(rng.integers(1, len(shape) + 1)),
                       clean_mask=iteration % 4 == 0, first_seen=iteration % 5 == 0,
                       weight_objective=(-1, 0, 1)[iteration % 3],
                       weight_mode=('min', 'max', 'mean', 'count', 'harmonic', 'mean_inv')[iteration % 6],
                       min_contact=1 + iteration % 3)
        snapshot = engine.prepare_labels(image, **options)
        expected = engine.label(image, n=8, max_depth=1, return_n=True, return_conflicts=True, **options)
        actual = snapshot.color(n=8, max_depth=1, return_n=True, return_conflicts=True, engine=engine)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(snapshot.color(n=8, max_depth=1, return_lut=True, engine=engine),
                                      engine.label(image, n=8, max_depth=1, return_lut=True, **options))
        counts['prepared_coloring'] += 1

    for _ in range(100):
        n = int(rng.integers(1, 22))
        edges = np.argwhere(np.triu(rng.random((n, n)) < rng.uniform(.05, .6), 1)).astype(np.int32)
        colors, used, conflicts = engine.color_graph(edges, n_vertices=n, n=4,
            max_depth=1, return_n=True, return_conflicts=True)
        assert conflicts == sum(colors[a] == colors[b] for a, b in edges)
        assert used == len(np.unique(colors)) and colors.min() > 0
        counts['graph_coloring'] += 1

    if include_geometry:
        geoms = [shapely.box(i + rng.uniform(0, .03), j, i + 1, j + 1)
                 for i in range(6) for j in range(5)]
        for tolerance in (0, .05):
            for threshold in (None, 0, .2):
                expected = set()
                for a in range(len(geoms)):
                    for b in range(a + 1, len(geoms)):
                        x, y = geoms[a], geoms[b]
                        if x.distance(y) > tolerance:
                            continue
                        if threshold is not None and (tolerance == 0 or threshold > 0):
                            length = (shapely.buffer(x, tolerance / 2).intersection(
                                shapely.buffer(y, tolerance / 2)).length / 2 if tolerance else x.intersection(y).length)
                            if length <= threshold:
                                continue
                        expected.add((a, b))
                assert pairs_of(geo.connect(geoms, tolerance=tolerance, min_shared_length=threshold)) == expected
                counts['geometry'] += 1

    jobs = [rng.integers(0, 5, (12, 13), dtype=np.int32) for _ in range(30)]
    def concurrent_check(image):
        out, count, conflicts = engine.label(image, n=8, expand=False,
            soft_conn=0, return_n=True, return_conflicts=True)
        assert conflicts == 0 and count == len(np.unique(out[out > 0]))
        np.testing.assert_array_equal(out == 0, image == 0)
        return 1
    with ThreadPoolExecutor(4) as pool:
        counts['concurrent_calls'] = sum(pool.map(concurrent_check, jobs))
    return dict(counts)


def fixed_workloads(include_geometry=True):
    rng = np.random.default_rng(21)
    engine = ncolor.Engine(n_threads=4)
    binary = (rng.random((512, 512)) < .7).astype(np.uint8)
    cells = (np.arange(2048, dtype=np.int32)[:, None] // 16 * 128
             + np.arange(2048, dtype=np.int32)[None, :] // 16 + 1)
    sparse = np.zeros((512, 512), np.int32)
    sparse[8::32, 8::32] = np.arange(1, 257).reshape(16, 16)
    labels = rng.choice(np.array([0, 2, 11, 100, 3000], np.int32), (1024, 1024))
    geoms = ([shapely.box(i * 1.01, j * 1.01, i * 1.01 + 1, j * 1.01 + 1)
              for i in range(35) for j in range(35)] if include_geometry else None)
    prepared = engine.prepare_labels(sparse)
    thin = (rng.random((2, 513, 517)) < .2).astype(np.uint8)
    byte_labels = labels.astype(np.uint8)
    workloads = {
        'prepared_color': lambda: prepared.color(engine=engine),
        'prepared_lookup': lambda: prepared.color(return_lut=True, engine=engine),
        'thin_components': lambda: engine.connected_components(thin, conn=1),
        'byte_format': lambda: engine.format_labels(byte_labels),
        'components': lambda: ncolor.connected_components(binary, conn=2),
        'components_singletons': lambda: ncolor.connected_components(binary.reshape((1,) * 8 + binary.shape), conn=2),
        'connect': lambda: engine.connect(cells),
        'format': lambda: engine.format_labels(labels),
        'expand_l1': lambda: engine.expand_labels(sparse, p=1),
        'expand_l2': lambda: engine.expand_labels(sparse, p=2),
        'label_standard': lambda: engine.label(sparse, expand_mode='standard'),
        'label_clean': lambda: engine.label(sparse, expand_mode='clean'),
    }
    if include_geometry:
        workloads['geometry'] = lambda: geo.connect(geoms, tolerance=.02, min_shared_length=.2)
    return engine, workloads


def resources(memory_blocks=50, include_geometry=True):
    engine, workloads = fixed_workloads(include_geometry)
    for _ in range(4):
        for call in workloads.values():
            call()
    timings = {}
    for name, call in workloads.items():
        samples = []
        for _ in range(15):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        timings[name] = {'median_ms': statistics.median(samples), 'samples_ms': samples}
    process = psutil.Process()
    samples, threads = [], []
    for block in range(memory_blocks):
        for _ in range(4):
            for call in workloads.values():
                call()
        if block % 2:
            engine.release_buffers()
        gc.collect()
        samples.append(process.memory_info().rss)
        threads.append(process.num_threads())
    # Worker allocator caches can warm late. Keep the complete history and
    # assess the final ten blocks, covering forty calls of each workload.
    growth = max(samples[-10:]) - min(samples[-10:])
    return {'timings': timings, 'memory': {'resident_bytes': samples,
            'post_warm_range_bytes': growth, 'thread_counts': threads,
            'measurement_blocks': 10, 'total_blocks': memory_blocks,
            'allowed_range_bytes': 8 * 1024 ** 2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-blocks', type=int, default=50)
    parser.add_argument('--skip-geometry', action='store_true',
                        help='omit optional geometry checks on hosts without that extra')
    args = parser.parse_args()
    if shapely is None and not args.skip_geometry:
        parser.error('Shapely is required unless --skip-geometry is explicit')
    if args.memory_blocks < 20:
        parser.error('--memory-blocks must be at least 20')
    result = {'seed': args.seed, 'platform': platform.system(),
              'architecture': platform.machine(), 'python': platform.python_version(),
              'numpy': np.__version__, 'geometry_included': not args.skip_geometry,
              'correctness': correctness(args.seed, not args.skip_geometry)}
    print('Correctness checks passed:', result['correctness'], flush=True)
    result.update(resources(args.memory_blocks, not args.skip_geometry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Memory range:', result['memory']['post_warm_range_bytes'], 'bytes', flush=True)
    print('Saved', args.output.resolve(), flush=True)
    memory = result['memory']
    assert memory['post_warm_range_bytes'] <= memory['allowed_range_bytes'], (
        'retained memory did not plateau', memory)
    assert len(set(memory['thread_counts'][-10:])) == 1, (
        'worker count grew', memory['thread_counts'])


if __name__ == '__main__':
    main()
