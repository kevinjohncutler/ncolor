"""ncolor — 4-color label graph coloring and label expansion utilities.

Public API resolves lazily (PEP 562 ``__getattr__``): bare ``import ncolor``
does almost no work; submodules — and their numba/C++ extension
dependencies — load only on first attribute access.

Default backend is the C++ engine in :mod:`ncolor._backend`. Set
``NCOLOR_BACKEND=numba`` in the environment to fall back to the original
numba reference implementation in :mod:`ncolor._numba_legacy` (kept as a
sanity-check toggle; will be removed once the C++ path is fully battle-
tested).

Public names:

* ``label``                — 4-color graph coloring of a label image
* ``unique_nonzero``       — unique nonzero labels
* ``format_labels``        — normalize labels to contiguous 1..N with bg=0
* ``get_lut``              — return the color lookup table built by ``label``
* ``expand_labels``        — Voronoi-style label expansion (L1 / L2)
* ``connected_components`` — N-D connected-components labelling
* ``regionprops``          — area / bbox / centroid for a labelled image
"""
import os as _os

from ._version import __version__  # cheap; no heavy deps

__all__ = [
    "label",
    "unique_nonzero",
    "format_labels",
    "get_lut",
    "expand_labels",
    "connected_components",
    "regionprops",
]

_BACKEND = _os.environ.get("NCOLOR_BACKEND", "cpp").lower()
if _BACKEND not in ("cpp", "numba"):
    raise ValueError(
        f"NCOLOR_BACKEND must be 'cpp' or 'numba', got {_BACKEND!r}"
    )

# Map public attribute -> source module (relative path).
if _BACKEND == "numba":
    _LAZY_ATTRS = {
        "label": "._numba_legacy.color",
        "unique_nonzero": "._numba_legacy.color",
        "get_lut": "._numba_legacy.color",
        "format_labels": ".format",
        "expand_labels": "._numba_legacy.expand",
    }
else:
    _LAZY_ATTRS = {
        "label": ".color",
        "unique_nonzero": ".color",
        "get_lut": ".color",
        "connected_components": ".color",
        "regionprops": ".color",
        "format_labels": ".format",
        "expand_labels": ".expand",
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
