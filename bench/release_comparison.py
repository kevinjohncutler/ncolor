"""Reproducible release and competitor timings with explicit semantics.

Generate shared inputs with ``corpus --output PATH``. Run ``run`` in a fresh
process per version and round, supplying its source directory, then use
``report`` to aggregate the JSON files. Keep historical builds isolated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import sysconfig
import time

# Also support python -S so unrelated editable-install hooks cannot select
# a different ncolor. Append dependencies after the standard library.
sys.path.append(sysconfig.get_path('purelib'))
import numpy as np


def fingerprint(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def corpus(output):
    from skimage.io import imread
    root = Path(__file__).resolve().parents[1]
    arrays = {}
    raw = imread(root / 'test_files/example.png')
    ys, xs = np.where(raw > 0)
    arrays['labels_logo'] = raw[ys.min():ys.max()+1, xs.min():xs.max()+1]
    arrays['labels_synthetic800'] = np.load(root / 'test_files/synthetic_800.npz')['labels']
    rng = np.random.default_rng(1709)
    for shape in [(1024, 1024), (2048, 2048), (96, 96, 96)]:
        image = np.zeros(shape, np.int32)
        count = max(50, image.size // (8000 if image.ndim == 2 else 20000))
        for label, center in enumerate(rng.integers(8, min(shape)-8, (count, len(shape))), 1):
            radius = int(rng.integers(3, 9))
            image[tuple(slice(int(x-radius), int(x+radius)) for x in center)] = label
        arrays['labels_' + 'x'.join(map(str, shape))] = image
    for name, image in list(arrays.items()):
        # Preserve background even if a fixture has no zero-valued pixels.
        values = np.unique(image[image != 0])
        arrays[name] = np.where(image == 0, 0, np.searchsorted(values, image) + 1).astype(np.int32)
    for shape in [(1024, 1024), (2048, 2048), (96, 96, 96), (2, 513, 517)]:
        for density in (.1, .7):
            arrays['mask_' + 'x'.join(map(str, shape)) + f'_{density}'] = rng.random(shape) < density
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(output.resolve(), flush=True)


def timed(call, validate, repeats):
    for _ in range(4):
        call()
    result = call()
    details = validate(result)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns()-start)/1e6)
    validate(result)
    return dict(samples_ms=samples, median_ms=statistics.median(samples), validation=details)


def run(args):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure, segmentation
    if args.version != 'external':
        sys.path.insert(0, str(args.source.resolve()))
        import ncolor
        assert Path(ncolor.__file__).resolve().is_relative_to(args.source.resolve())
        engine = ncolor.Engine(n_threads=args.threads) if hasattr(ncolor, 'Engine') else None
    else:
        ncolor = engine = None
    version1 = args.version == '1.5.3'
    inputs = np.load(args.corpus)
    results = {}
    def record(name, fn, check):
        results[name] = timed(fn, check, args.repeats)
        print(name, round(results[name]['median_ms'], 3), flush=True)
    for name in inputs.files:
        image = inputs[name]
        original = fingerprint(image)
        if name.startswith('labels_'):
            if ncolor is not None:
                for mode in ('default', 'matched'):
                    options = dict(n=4, max_depth=30, return_n=True)
                    if mode == 'matched':
                        options.update(conn=1, format_input=False)
                        if not version1:
                            options.update(expand_mode='standard', soft_conn=0, soft_radius=0)
                    label = ncolor.label if version1 else engine.label
                    def check_coloring(result):
                        colors, used = result
                        mask_errors = int(np.count_nonzero((colors == 0) != (image == 0)))
                        if not version1: assert mask_errors == 0
                        assert int(used) == len(np.unique(colors[colors > 0]))
                        checked = label(image, return_conflicts=True, **options)
                        assert checked[-1] == 0, ('coloring conflicts', name, mode, checked[-1])
                        return dict(fingerprint=fingerprint(colors), colors=int(used), conflicts=0,
                                    foreground_errors=mask_errors, valid=mask_errors == 0)
                    record(name+'/'+mode, lambda: label(image, **options), check_coloring)
            if ncolor is None:
                def feature_transform():
                    indices = ndimage.distance_transform_edt(image == 0, return_distances=False, return_indices=True)
                    return image[tuple(indices)]
                expansions = {'scipy_feature': feature_transform,
                              'skimage_expand': lambda: segmentation.expand_labels(image, distance=np.inf)}
            else:
                expansions = {'expand': lambda: (ncolor.expand_labels(image) if version1 else engine.expand_labels(image, p=2))}
            expected_indices = ndimage.distance_transform_edt(image == 0, return_distances=False, return_indices=True)
            expected = image[tuple(expected_indices)]
            def check_expansion(result):
                np.testing.assert_array_equal(result[image != 0], image[image != 0])
                assert not np.any(result == 0)
                assert np.isin(result, np.unique(image)).all()
                # Check every differing pixel against the nearest source of
                # its chosen label. Different output labels must be exact ties.
                different = np.argwhere(result != expected)
                if different.size:
                    coords = np.argwhere(image != 0)
                    source_ids = image[tuple(coords.T)]
                    chosen_ids = result[tuple(different.T)]
                    nearest = expected_indices[(slice(None), *different.T)].T
                    expected_d2 = ((different-nearest)**2).sum(axis=1)
                    for chosen in np.unique(chosen_ids):
                        selected = chosen_ids == chosen
                        tree = cKDTree(coords[source_ids == chosen])
                        distances, _ = tree.query(different[selected])
                        np.testing.assert_allclose(distances**2, expected_d2[selected], rtol=1e-12, atol=1e-12)
                return dict(fingerprint=fingerprint(result), differing_tie_pixels=int(np.count_nonzero(result != expected)))
            for op, call in expansions.items():
                record(name+'/'+op, call, check_expansion)
            if not version1:
                reference = measure.regionprops_table(image, properties=('area', 'bbox', 'centroid'))
                def properties():
                    if ncolor is not None:
                        return ncolor.regionprops(image)
                    regions = measure.regionprops(image)
                    return dict(area=np.array([r.area for r in regions]),
                                bbox_min=np.array([r.bbox[:image.ndim] for r in regions]),
                                bbox_max=np.array([r.bbox[image.ndim:] for r in regions]),
                                centroid=np.array([r.centroid for r in regions]))
                def check_properties(result):
                    np.testing.assert_array_equal(result['area'], reference['area'])
                    for axis in range(image.ndim):
                        np.testing.assert_array_equal(result['bbox_min'][:,axis], reference[f'bbox-{axis}'])
                        np.testing.assert_array_equal(result['bbox_max'][:,axis], reference[f'bbox-{axis+image.ndim}'])
                        np.testing.assert_allclose(result['centroid'][:,axis], reference[f'centroid-{axis}'], rtol=0, atol=1e-12)
                    return dict(regions=len(result['area']), fingerprint=fingerprint(result['area']))
                record(name+'/properties', properties, check_properties)
                if ncolor is None:
                    def table_properties():
                        return measure.regionprops_table(image, properties=('area', 'bbox', 'centroid'))
                    def check_table(result):
                        for key, values in reference.items():
                            np.testing.assert_allclose(result[key], values, rtol=0, atol=1e-12)
                        return dict(regions=len(result['area']), fingerprint=fingerprint(result['area']))
                    record(name+'/properties_table', table_properties, check_table)
        elif not version1:
            for conn in sorted({1, image.ndim}):
                expected, count = measure.label(image, connectivity=conn, return_num=True)
                def components():
                    if ncolor is None: return measure.label(image, connectivity=conn, return_num=True)
                    if args.version == '2.2.0': return ncolor.connected_components(image, conn=conn)
                    return engine.connected_components(image, conn=conn)
                def check_components(result):
                    labels, actual_count = result
                    assert actual_count == count
                    np.testing.assert_array_equal(labels, expected)
                    return dict(components=int(count), fingerprint=fingerprint(labels))
                record(name+f'/components_c{conn}', components, check_components)
        assert fingerprint(image) == original
    metadata = dict(version=args.version, revision=args.revision, round=args.round,
                    threads=args.threads, repeats=args.repeats, warmups=4,
                    python=platform.python_version(), platform=platform.system(), machine=platform.machine(),
                    cpu=(subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
                         if platform.system() == 'Darwin' else platform.processor() or platform.machine()),
                    dependencies={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-image','numba','fastremap']},
                    corpus={k:fingerprint(inputs[k]) for k in inputs.files})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(metadata=metadata,results=results),indent=2)+'\n')
    print(args.output.resolve(),flush=True)


def report(args):
    grouped = {}
    validations = {}
    corpus_ids = None
    environment = None
    revisions = {}
    for path in sorted(args.directory.glob('round_*.json')):
        data = json.loads(path.read_text())
        version = data['metadata']['version']
        meta = data['metadata']
        signature = {key:meta[key] for key in ('cpu','python','dependencies','threads','repeats')}
        if environment is None: environment = signature
        assert environment == signature, 'benchmark environments differ'
        assert revisions.setdefault(version, meta['revision']) == meta['revision'], 'source revisions differ'
        if corpus_ids is None: corpus_ids = data['metadata']['corpus']
        assert corpus_ids == data['metadata']['corpus'], 'input corpus differs between runs'
        for name, result in data['results'].items():
            grouped.setdefault(name,{}).setdefault(version,[]).append(result['median_ms'])
            validations.setdefault(name,{}).setdefault(version,[]).append(result['validation'])
    summary = {}
    for name, versions in grouped.items():
        medians={v:statistics.median(samples) for v,samples in versions.items()}
        summary[name]=dict(median_ms=medians,round_medians_ms=versions, validation=validations[name],
                           speedup_vs_current={v:t/medians['current'] for v,t in medians.items()
                                               if v!='current' and all(x.get('valid',True) for x in validations[name][v])
                                               and all(x.get('valid',True) for x in validations[name]['current'])}
                                               if 'current' in medians else {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(args.output.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('corpus');make.add_argument('--output',type=Path,required=True)
    worker=sub.add_parser('run')
    worker.add_argument('--source',type=Path)
    worker.add_argument('--version',choices=['1.5.3','2.2.0','current','external'],required=True)
    worker.add_argument('--revision',default='external')
    worker.add_argument('--corpus',type=Path,required=True)
    worker.add_argument('--output',type=Path,required=True)
    worker.add_argument('--threads',type=int,default=4)
    worker.add_argument('--round',type=int,required=True)
    worker.add_argument('--repeats',type=int,default=15)
    summarize=sub.add_parser('report');summarize.add_argument('--directory',type=Path,required=True);summarize.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='corpus': corpus(args.output)
    elif args.command=='run':
        if args.version != 'external' and args.source is None:
            parser.error('--source is required for ncolor runs')
        if args.repeats < 1 or args.threads < 1:
            parser.error('--repeats and --threads must be positive')
        run(args)
    else: report(args)


if __name__=='__main__': main()
