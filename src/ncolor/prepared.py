"""Owned contact graphs for repeated coloring of an unchanged label image."""
from __future__ import annotations

import numpy as np

from ._engines import _use
from .color import label, _validate_coloring_budget


class PreparedLabels:
    """Immutable raster topology and output-label mapping.

    Create with :func:`prepare_labels`. The snapshot owns its data and is
    independent of subsequent input mutations, engine calls, or buffer
    release. Recreate it to change the image, connectivity, expansion,
    cleanup, contact filtering, or weight reducer. Multiple engines can
    color the same snapshot concurrently.
    """
    __slots__ = ('__data',)

    def __init__(self, data):
        self.__data = data

    @property
    def shape(self):
        return tuple(self.__data.shape)

    @property
    def n_labels(self):
        """Number of graph vertices, including isolated labels."""
        return self.__data.n_labels

    @property
    def nbytes(self):
        """Bytes of owned array data, excluding container overhead."""
        return self.__data.nbytes

    def color(self, n=4, max_depth=30, *, de_table=None, out=None,
              return_n=False, return_lut=False, check_conflicts=False,
              return_conflicts=False, engine=None):
        """Recolor the snapshot using the same picker as :func:`label`.

        Color target, search depth, and perceptual palette can change on
        each call. Topology and weight settings are fixed at preparation.
        Return flags and ``out`` follow :func:`label`; ``return_lut=True``
        returns the label-to-color lookup table instead of the image. This
        skips image allocation and rendering unless ``out`` is supplied.
        """
        n, max_depth = _validate_coloring_budget(n, max_depth)
        palette = None if de_table is None else np.ascontiguousarray(de_table, dtype=np.float64)
        with _use(engine) as current:
            solver = current._solver
            image, used = solver.color_prepared(
                self.__data, n_colors=n, max_depth=max_depth,
                de_table=palette, out=out, render=not return_lut)
            result = solver.get_last_lut() if return_lut else image
            conflicts = solver.get_last_n_conflicts() if (check_conflicts or return_conflicts) else 0
        if check_conflicts and conflicts:
            raise ValueError(f'Coloring conflict detected: {conflicts} adjacent pairs share a color.')
        if return_n and return_conflicts:
            return result, int(used), conflicts
        if return_n:
            return result, int(used)
        if return_conflicts:
            return result, conflicts
        return result


def prepare_labels(lab, *, conn=1, expand=True, format_input=True,
                   p=2, wrap=False, first_seen=False, weight_objective=0,
                   weight_mode='min', extra_edges=None, connect_radius=1,
                   min_contact=1, expand_mode='clean', soft_extra_edges=None,
                   soft_conn=2, soft_radius=2, clean_mask=False, _engine=None):
    """Prepare an owned snapshot for repeated raster coloring.

    All options have the same meaning as in :func:`label`. Preparation
    performs normalization, expansion, contact extraction, and graph
    construction once, without solving a coloring. It retains compact label
    identifiers plus the hard and soft graph data. Large sparse snapshots
    store only foreground positions and labels; empty snapshots need no
    per-pixel map. Identifiers use one, two, or four bytes according to label
    count. Source arrays and explicit edge lists are not retained by reference.

    >>> prepared = prepare_labels([[0, 1, 1], [2, 2, 0]])
    >>> image = prepared.color(n=4)
    >>> image.shape
    (2, 3)
    """
    from ._backend import _impl
    data = _impl.PreparedRaster()
    label(lab, conn=conn, expand=expand, format_input=format_input,
          p=p, wrap=wrap, first_seen=first_seen, weight_objective=weight_objective,
          weight_mode=weight_mode, extra_edges=extra_edges,
          connect_radius=connect_radius, min_contact=min_contact,
          expand_mode=expand_mode, soft_extra_edges=soft_extra_edges,
          soft_conn=soft_conn, soft_radius=soft_radius, clean_mask=clean_mask,
          _engine=_engine, _prepared=data)
    return PreparedLabels(data)
