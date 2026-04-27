"""ncolor — 4-color label graph coloring and label expansion utilities.

Public API resolves lazily (PEP 562 ``__getattr__``): bare ``import ncolor``
does almost no work; submodules — and their numba/scipy dependencies —
load only on first attribute access. This matters because ``_label`` pulls
in numba JIT-decorated functions whose @njit cache lookups dominate
cold-start time on NAS-mounted source trees.

The implementation submodules are private (``_label``, ``_format_labels``,
``_expand_labels``) so their names don't shadow the public function names
they provide. The previous public-name modules (``label.py`` etc.) were
renamed to free up the namespace — the public API at ``ncolor.<name>`` is
unchanged.

Public names:

* ``label``          — 4-color graph coloring of a label image
* ``unique_nonzero`` — unique nonzero labels (uses fastremap.unique)
* ``format_labels``  — normalize labels to contiguous 1..N with bg=0
* ``get_lut``        — return the color lookup table built by ``label``
* ``expand_labels``  — multi-pass label expansion across background pixels
"""

from ._version import __version__  # cheap; no heavy deps

__all__ = [
    "label",
    "unique_nonzero",
    "format_labels",
    "get_lut",
    "expand_labels",
]

# Map public attribute -> source module (relative path).
_LAZY_ATTRS = {
    "label": "._label",
    "unique_nonzero": "._label",
    "get_lut": "._label",
    "format_labels": "._format_labels",
    "expand_labels": "._expand_labels",
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
