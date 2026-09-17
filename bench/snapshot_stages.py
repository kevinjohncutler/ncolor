"""Measure actual preparation stages to guide the next optimization round."""
import argparse
import json
from pathlib import Path
import statistics
import sys

from release_comparison import cpu_model, fingerprint
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
import ncolor
from ncolor._backend import _impl
from ncolor.color import label
assert Path(ncolor.__file__).resolve().is_relative_to(args.source.resolve())
engine = ncolor.Engine(n_threads=4)
shape = (2048, 2048)
seeds = np.zeros(shape, np.int32)
view = seeds[8::32, 8::32]
view[:] = np.arange(1, view.size+1).reshape(view.shape)
dense = engine.expand_labels(seeds)
rng = np.random.default_rng(802)
results = {}
for fraction in (.001, .01, .1, 1.):
    image = np.where(rng.random(shape) < fraction, dense, 0).astype(np.int32)
    stages = {}
    for rep in range(19):
        prepared = _impl.PreparedRaster()
        label(image, _engine=engine, _prepared=prepared, verbose=True)
        if rep >= 4:
            for name, ms in engine._solver.get_last_stages():
                stages.setdefault(name, []).append(ms)
    expected = engine.label(image, n=32)
    actual, _ = engine._solver.color_prepared(prepared, n_colors=32)
    np.testing.assert_array_equal(actual, expected)
    results[str(fraction)] = dict(
        stage_median_ms={key: statistics.median(values) for key, values in stages.items()},
        samples_ms=stages, snapshot_bytes=prepared.nbytes, fingerprint=fingerprint(actual))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=4, shape=shape,
                                     results=results), indent=2)+'\n')
print(args.output.resolve())
