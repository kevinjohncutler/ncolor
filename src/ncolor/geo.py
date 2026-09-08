"""Vector-geometry front end: 4-color polygons instead of pixels.

``ncolor.label`` colors a *raster* label image. This module colors
vector features directly: a GeoPandas ``GeoDataFrame`` / ``GeoSeries``,
a GeoJSON file, string or ``dict``, a list of Shapely geometries, or
anything exposing ``__geo_interface__``. Data that starts out as
polygons never has to round-trip through a raster to get a coloring.

    import geopandas as gpd
    from ncolor import geo

    gdf = gpd.read_file("cells.geojson")
    gdf["color"] = geo.label(gdf)

Adjacency comes from geometry rather than a pixel walk; the coloring
itself is the same C++ picker ``ncolor.label`` uses, reached through
:func:`ncolor.color_graph`.

Two contact rules matter, and both have raster analogues:

* **Point contacts don't count** (default). Regions meeting at a single
  point (the "Four Corners" case) are not adjacent. This mirrors
  ``conn=1`` (face-only) on the raster side, and it is what keeps the
  adjacency graph planar, hence 4-colorable. Pass
  ``min_shared_length=None`` to count them anyway (expect n=5+).
* **A tolerance closes gaps.** Polygons vectorized from a raster, or
  reprojected, often end up with hairline gaps where they should touch.
  ``tolerance=eps`` treats features within ``eps`` of each other as
  adjacent, the analogue of ``connect_radius`` on the raster side.

Requires Shapely 2.0+ (``pip install ncolor[geo]``). GeoPandas is
optional and used only when the input or requested output is one of its
types.
"""
from __future__ import annotations

import os

import numpy as np

from .color import color_graph

__all__ = ["connect", "label"]


def _require_shapely():
    try:
        import shapely
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "ncolor's geometry support requires Shapely 2.0+. "
            "Install it with: pip install 'ncolor[geo]'"
        ) from exc
    if not hasattr(shapely, "STRtree"):             # pragma: no cover
        raise ImportError(
            f"ncolor's geometry support requires Shapely 2.0+; found "
            f"{getattr(shapely, '__version__', 'unknown')}")
    return shapely


def _geoms_from_geojson(obj, shapely):
    """Extract a geometry list from a parsed GeoJSON mapping."""
    from shapely.geometry import shape

    kind = obj.get("type")
    if kind == "FeatureCollection":
        return [None if f.get("geometry") is None else shape(f["geometry"])
                for f in obj.get("features", [])]
    if kind == "Feature":
        geom = obj.get("geometry")
        return [None if geom is None else shape(geom)]
    if kind == "GeometryCollection":
        return [shape(g) for g in obj.get("geometries", [])]
    if kind is None:
        raise ValueError(
            "mapping does not look like GeoJSON (no 'type' key)")
    return [shape(obj)]


def _as_geometries(geoms):
    """Normalize any supported input to (object ndarray of geoms, frame).

    ``frame`` is the original GeoDataFrame/GeoSeries when the input was
    one, else None. Kept so ``return_frame=True`` can hand back
    something with the caller's other columns intact.
    """
    shapely = _require_shapely()
    from shapely.geometry.base import BaseGeometry

    frame = None

    # GeoPandas types, detected by duck-typing so geopandas stays an
    # optional dependency (it is never imported here).
    if hasattr(geoms, "geometry") and hasattr(geoms, "crs"):
        frame = geoms
        geoms = geoms.geometry
    if hasattr(geoms, "to_numpy") and hasattr(geoms, "crs"):
        if frame is None:
            frame = geoms
        arr = np.asarray(geoms.to_numpy(), dtype=object)
        return arr, frame

    # A path or a raw GeoJSON string.
    if isinstance(geoms, (str, os.PathLike)):
        text = str(geoms)
        if not text.lstrip().startswith("{"):
            try:
                import geopandas as gpd
            except ImportError:
                import json
                with open(geoms, "r", encoding="utf-8") as fh:
                    return _as_geometries(json.load(fh))
            frame = gpd.read_file(geoms)
            return np.asarray(frame.geometry.to_numpy(), dtype=object), frame
        import json
        return _as_geometries(json.loads(text))

    # GeoJSON mapping, or any object publishing __geo_interface__
    # (Fiona features, GeoPandas objects on older versions, ...).
    if isinstance(geoms, dict):
        return np.asarray(_geoms_from_geojson(geoms, shapely), dtype=object), None
    if hasattr(geoms, "__geo_interface__") and not hasattr(geoms, "__len__"):
        return _as_geometries(geoms.__geo_interface__)

    if isinstance(geoms, BaseGeometry):
        return np.asarray([geoms], dtype=object), None

    # A DataFrame-shaped object reaching this point has no geometry we
    # can use: either a plain pandas DataFrame, or a GeoDataFrame whose
    # active geometry column was never set (``.geometry`` raises there, so
    # the duck-typed check above declined it). Iterating one yields column
    # *names*, so without this the failure is "cannot interpret str as a
    # geometry", which points nowhere near the cause.
    if hasattr(geoms, "columns") and hasattr(geoms, "index"):
        raise TypeError(
            "no active geometry column on this DataFrame; set one with "
            "gdf.set_geometry('<column>'), or pass the geometries directly")

    # Any iterable of geometries / GeoJSON mappings / __geo_interface__.
    from shapely.geometry import shape
    out = []
    for g in geoms:
        if g is None or isinstance(g, BaseGeometry):
            out.append(g)
        elif isinstance(g, dict):
            out.append(shape(g.get("geometry", g)))
        elif hasattr(g, "__geo_interface__"):
            gi = g.__geo_interface__
            out.append(shape(gi.get("geometry", gi)))
        else:
            raise TypeError(
                f"cannot interpret {type(g).__name__} as a geometry; pass "
                f"Shapely geometries, GeoJSON mappings, or objects with "
                f"__geo_interface__")
    return np.asarray(out, dtype=object), None


def _candidate_pairs(shapely, geoms, tolerance):
    """Unique (lo, hi) index pairs whose geometries are within tolerance."""
    if len(geoms) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    tree = shapely.STRtree(geoms)
    if tolerance > 0:
        hits = tree.query(geoms, predicate="dwithin", distance=float(tolerance))
    else:
        hits = tree.query(geoms, predicate="intersects")
    left, right = hits[0], hits[1]
    keep = left < right                             # drops self-hits + mirrors
    return np.stack([left[keep], right[keep]], axis=1).astype(np.int32, copy=False)


def _contact_lengths(shapely, geoms, pairs, tolerance):
    """Length of the shared boundary for each pair.

    With ``tolerance > 0`` the geometries are dilated by half the
    tolerance first, so features separated by a hairline gap still
    report the length of the seam that would close it.
    """
    a = geoms[pairs[:, 0]]
    b = geoms[pairs[:, 1]]
    if tolerance > 0:
        half = float(tolerance) / 2.0
        a = shapely.buffer(a, half)
        b = shapely.buffer(b, half)
        return shapely.length(shapely.intersection(a, b)) / 2.0
    return shapely.length(shapely.intersection(a, b))


def connect(geoms, tolerance=0.0, min_shared_length=0.0,
            return_rejected=False):
    """Adjacency pairs for a set of vector features.

    The vector counterpart of :func:`ncolor.connect`: instead of walking
    pixels it queries an R-tree over the geometries and keeps the pairs
    that actually share a boundary.

    Parameters
    ----------
    geoms : GeoDataFrame, GeoSeries, GeoJSON dict/str/path, or sequence
        Features to test. See the module docstring for the accepted
        forms. Missing (``None``) and empty geometries take part in no
        pairs.
    tolerance : float
        Treat features within this distance of each other as touching.
        0 (default) requires exact contact. Use a small positive value
        for polygons vectorized from a raster or reprojected, where
        neighbors are often separated by a hairline gap. In the units of
        the data's CRS (Coordinate Reference System).
    min_shared_length : float or None
        Minimum shared-boundary length for a pair to count as adjacent,
        compared strictly (``length > min_shared_length``). The default
        of 0.0 drops point-only contacts, the "Four Corners" rule that
        keeps a planar partition 4-colorable. ``None`` disables the
        filter, so any contact at all counts (and the graph may then
        need 5+ colors). Only applied to pairs of areal geometries;
        lines and points are never filtered by it. With a ``tolerance``,
        contact is measured on the geometries dilated by half of it, and
        a threshold of 0 is skipped as a no-op.
    return_rejected : bool
        Also return the candidate pairs the filter rejected. These are
        the natural *soft* constraints: near-misses that should still
        differ in color when it's free to do so.

    Returns
    -------
    pairs : (M, 2) int32 ndarray
        Unique ``(lo, hi)`` index pairs, **0-indexed by position** in the
        input (unlike :func:`ncolor.connect`, which returns 1-indexed
        label IDs).
    rejected : (K, 2) int32 ndarray
        Only when ``return_rejected=True``.
    """
    shapely = _require_shapely()
    geoms, _ = _as_geometries(geoms)
    return _connect_arr(shapely, geoms, tolerance, min_shared_length,
                        return_rejected)


def _connect_arr(shapely, geoms, tolerance, min_shared_length, return_rejected):
    """connect() body, on an already-normalized geometry array."""
    pairs = _candidate_pairs(shapely, geoms, tolerance)

    rejected = np.zeros((0, 2), dtype=np.int32)
    # A zero threshold across a tolerance gap would reject nothing (every
    # candidate is by definition in contact once dilated), and measuring
    # it would buffer every geometry in every candidate pair. Skip it.
    filter_active = (min_shared_length is not None
                     and (tolerance <= 0 or min_shared_length > 0))
    if filter_active and len(pairs):
        # The length rule is meaningful only where a shared *boundary*
        # exists, i.e. between areal geometries; a line/line or
        # point/line contact is legitimately zero-length. area() is nan
        # for a missing geometry, which is not areal either.
        areal = np.nan_to_num(shapely.area(geoms), nan=0.0) > 0
        testable = areal[pairs[:, 0]] & areal[pairs[:, 1]]
        if testable.any():
            lengths = _contact_lengths(shapely, geoms, pairs[testable], tolerance)
            ok = np.ones(len(pairs), dtype=bool)
            ok[testable] = lengths > float(min_shared_length)
            rejected = pairs[~ok]
            pairs = pairs[ok]

    if return_rejected:
        return pairs, rejected
    return pairs


def label(geoms, n=4, tolerance=0.0, min_shared_length=0.0, soft=True,
          return_n=False, return_frame=False, column="color",
          check_conflicts=False, return_conflicts=False):
    """4-color a set of vector features.

    Assigns each feature a color in ``1..n`` such that features sharing
    a boundary get different colors. It is the vector counterpart of
    :func:`ncolor.label`, with no rasterization step, so the input
    geometry is preserved exactly.

        import geopandas as gpd
        from ncolor import geo
        gdf = gpd.read_file("cells.geojson")
        gdf["color"] = geo.label(gdf)
        gdf.plot(column="color", cmap="viridis")

    Parameters
    ----------
    geoms : GeoDataFrame, GeoSeries, GeoJSON dict/str/path, or sequence
        Features to color. See the module docstring for accepted forms.
    n : int
        Color target. A planar partition of the plane always fits in 4,
        but overlapping features, or ``min_shared_length=None`` (point
        contacts count), can force more; such a graph escalates past the
        target rather than coming back with a broken coloring, so read
        the count back with ``return_n=True`` if it matters.
    tolerance, min_shared_length :
        Adjacency rules, documented on :func:`ncolor.geo.connect`.
    soft : bool
        Feed the pairs rejected by ``min_shared_length`` to the
        soft-constraint pass, so near-miss neighbors still get different
        colors where that's possible without breaking the hard coloring.
        No effect when nothing was rejected.
    return_frame : bool
        Return a GeoDataFrame (requires GeoPandas) with the colors in
        ``column`` instead of a bare array. When the input was already a
        GeoDataFrame its other columns are carried over.
    column : str
        Column name used by ``return_frame``. A column of this name
        already on the input frame is replaced in the returned copy; the
        input itself is never modified.

    Returns
    -------
    colors : (n_features,) uint8 ndarray
        Color per feature, in ``1..n_used``, in input order. Missing and
        empty geometries get 0, matching the raster convention where 0
        means background.
    """
    shapely = _require_shapely()
    geoms, frame = _as_geometries(geoms)
    n_geoms = len(geoms)

    pairs, rejected = _connect_arr(shapely, geoms, tolerance,
                                   min_shared_length, return_rejected=True)
    result = color_graph(pairs, n_vertices=n_geoms, n=n,
                         soft_edges=rejected if (soft and len(rejected)) else None,
                         return_n=True, check_conflicts=check_conflicts,
                         return_conflicts=return_conflicts)
    if return_conflicts:
        colors, n_used, conflicts = result
    else:
        colors, n_used = result

    if n_geoms:
        # 0 for missing / empty geometries, mirroring label()'s background.
        blank = shapely.is_missing(geoms) | shapely.is_empty(geoms)
        if blank.any():
            colors = colors.copy()
            colors[blank] = 0

    out = colors
    if return_frame:
        try:
            import geopandas as gpd
        except ImportError as exc:                  # pragma: no cover
            raise ImportError(
                "return_frame=True requires GeoPandas: pip install geopandas"
            ) from exc
        if frame is not None and hasattr(frame, "assign"):
            out = frame.copy()                       # a GeoDataFrame
            out[column] = colors
        else:
            # A GeoSeries (which has no .assign) or a non-GeoPandas input.
            # Carry the CRS and the index across when there was a frame to
            # take them from; a bare geometry list has neither.
            out = gpd.GeoDataFrame({column: colors}, geometry=list(geoms),
                                   crs=getattr(frame, "crs", None),
                                   index=getattr(frame, "index", None))

    if return_n and return_conflicts:
        return out, int(n_used), conflicts
    if return_n:
        return out, int(n_used)
    if return_conflicts:
        return out, conflicts
    return out
