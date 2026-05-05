"""Legacy numba implementation of ncolor's connect / expand / label.

Kept as a fallback toggle (``NCOLOR_BACKEND=numba``) so callers can
sanity-check the C++ engine against the original numba reference without
reinstalling. Default backend is the C++ engine in :mod:`ncolor._backend`.
"""
