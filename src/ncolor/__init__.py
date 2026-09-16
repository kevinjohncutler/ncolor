"""ncolor — 4-color label graph coloring and label expansion utilities.

The public API resolves lazily via PEP 562 ``__getattr__``: bare
``import ncolor`` does almost no work; submodules and the C++
extension load on first attribute access.

Public names:

* ``prepare_labels``: create an owned snapshot for repeated coloring
* ``PreparedLabels``: snapshot with reusable contacts and output labels

* ``label``                — 4-color graph coloring of a label image
* ``connect``              — adjacency pairs in a label image
* ``format_labels``        — normalize labels to contiguous 1..N with bg=0
* ``expand_labels``        — Voronoi-style label expansion (L1 / L2)
* ``connected_components`` — N-D connected-components labeling
* ``regionprops``          — area / bbox / centroid for a labeled image
* ``delete_spurs``         — N-D skeleton hole-fill + endpoint pruning
* ``color_graph``          — 4-color an abstract graph from an edge list
* ``geo``                  — vector front end: ``geo.label`` / ``geo.connect``
                             for GeoDataFrames, GeoJSON and Shapely geometries
* ``release_buffers``      — free the scratch memory kept between calls
* ``Engine``               — an independent engine, for coloring several
                             images at once from different threads
"""
from ._version import __version__

__all__ = [
    "label",
    "connect",
    "format_labels",
    "expand_labels",
    "connected_components",
    "regionprops",
    "delete_spurs",
    "color_graph",
    "geo",
    "release_buffers",
    "Engine",
    "prepare_labels",
    "PreparedLabels",
]

_LAZY_ATTRS = {
    "label": ".color",
    "connect": ".color",
    "connected_components": ".color",
    "regionprops": ".color",
    "format_labels": ".format",
    "expand_labels": ".expand",
    "delete_spurs": ".format",
    "color_graph": ".color",
    "release_buffers": "._engines",
    "Engine": "._engines",
    "prepare_labels": ".prepared",
    "PreparedLabels": ".prepared",
}

# Submodules reachable as a plain attribute (``ncolor.geo.label``) after
# a bare ``import ncolor``. Without this hook that access raises
# AttributeError until something has imported the submodule; ``import
# ncolor.geo`` and ``from ncolor import geo`` work either way. ``geo``
# itself imports shapely lazily, inside its functions, so touching the
# attribute stays as cheap as the rest of the package.
_LAZY_SUBMODULES = {"geo"}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
