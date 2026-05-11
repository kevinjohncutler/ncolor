"""ncolor — 4-color label graph coloring and label expansion utilities.

The public API resolves lazily via PEP 562 ``__getattr__``: bare
``import ncolor`` does almost no work; submodules and the C++
extension load on first attribute access.

Public names:

* ``label``                — 4-color graph coloring of a label image
* ``connect``              — adjacency pairs in a label image
* ``format_labels``        — normalize labels to contiguous 1..N with bg=0
* ``get_lut``              — return the color lookup table built by ``label``
* ``expand_labels``        — Voronoi-style label expansion (L1 / L2)
* ``connected_components`` — N-D connected-components labelling
* ``regionprops``          — area / bbox / centroid for a labelled image
* ``delete_spurs``         — N-D skeleton hole-fill + endpoint pruning
"""
from ._version import __version__

__all__ = [
    "label",
    "connect",
    "format_labels",
    "get_lut",
    "expand_labels",
    "connected_components",
    "regionprops",
    "delete_spurs",
]

_LAZY_ATTRS = {
    "label": ".color",
    "connect": ".color",
    "get_lut": ".color",
    "connected_components": ".color",
    "regionprops": ".color",
    "format_labels": ".format",
    "expand_labels": ".expand",
    "delete_spurs": ".format",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
