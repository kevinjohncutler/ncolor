"""Measure repeated raster coloring with fixed contact topology.

This experiment uses existing graph-coloring calls. Geometry and contact
settings must remain unchanged; rebuilding is required when they change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import ncolor

from feature_experiments import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    engine = ncolor.Engine(n_threads=4)
    image = np.zeros((2048, 2048), np.int32)
    image[8::32, 8::32] = np.arange(1, 4097).reshape(64, 64)
    expanded = engine.expand_labels(image, mode='clean')
    edges = engine.connect(expanded) - 1
    expected = engine.label(image)
    soft = engine._solver.get_last_soft_pairs() - 1

    def prepared():
        colors = engine.color_graph(edges, n_vertices=int(image.max()), soft_edges=soft)
        lut = np.concatenate((np.zeros(1, np.uint8), colors))
        return lut[image]

    np.testing.assert_array_equal(prepared(), expected)
    result = {'complete_pipeline': measure(lambda: engine.label(image), 25),
              'reuse_contacts': measure(prepared, 25),
              'hard_edges': len(edges), 'soft_edges': len(soft)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
