"""Recheck property timings without coloring worker pools or scheduler migration.

Use the same CPU affinity for each source version on systems that support it.
Run in alternating fresh processes; validation is outside the measured calls.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from release_comparison import cpu_model, fingerprint, timed
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import ncolor
    from skimage.measure import regionprops_table
    assert Path(ncolor.__file__).resolve().is_relative_to(args.source.resolve())
    results = {}
    with np.load(args.corpus) as inputs:
        for name in inputs.files:
            if not name.startswith('labels_'):
                continue
            image = inputs[name]
            reference = regionprops_table(image, properties=('area', 'bbox', 'centroid'))

            def check(result):
                np.testing.assert_array_equal(result['area'], reference['area'])
                for axis in range(image.ndim):
                    np.testing.assert_array_equal(result['bbox_min'][:, axis], reference[f'bbox-{axis}'])
                    np.testing.assert_array_equal(result['bbox_max'][:, axis], reference[f'bbox-{axis+image.ndim}'])
                    np.testing.assert_allclose(result['centroid'][:, axis], reference[f'centroid-{axis}'], atol=1e-12, rtol=0)
                return {'input': fingerprint(image), 'area': fingerprint(result['area'])}

            results[name] = timed(lambda: ncolor.regionprops(image), check, 101)
    output = dict(version=args.version, cpu=cpu_model(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
