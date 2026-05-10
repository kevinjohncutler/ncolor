"""ncolor — 4-color label graph coloring and label expansion utilities.

Public API resolves lazily (PEP 562 ``__getattr__``): bare ``import ncolor``
does almost no work; submodules — and the C++ extension — load only on
first attribute access.

The original numba reference implementation was retired in v1.6.0; the
C++ engine in :mod:`ncolor._backend` is the only backend.

Public names:

* ``label``                — 4-color graph coloring of a label image
* ``format_labels``        — normalize labels to contiguous 1..N with bg=0
* ``get_lut``              — return the color lookup table built by ``label``
* ``expand_labels``        — Voronoi-style label expansion (L1 / L2)
* ``connected_components`` — N-D connected-components labelling
* ``regionprops``          — area / bbox / centroid for a labelled image
* ``delete_spurs``         — N-D skeleton hole-fill + endpoint pruning

The pure-numpy helpers ``unique_nonzero`` and ``is_sequential`` were
removed in v2 — they aren't called from any ncolor hot path; identical
implementations now live in ``ocdkit.array.ops``.
"""
from ._version import __version__  # cheap; no heavy deps

__all__ = [
    "label",
    "format_labels",
    "get_lut",
    "expand_labels",
    "connected_components",
    "regionprops",
    "delete_spurs",
]

# Map public attribute -> source module (relative path).
_LAZY_ATTRS = {
    "label": ".color",
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
