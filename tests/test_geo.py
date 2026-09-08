"""Tests for the vector-geometry front end (``ncolor.geo``)."""
import json
import os

import numpy as np
import pytest

import ncolor
from ncolor import geo

shapely = pytest.importorskip("shapely", minversion="2.0")
from shapely import LineString, MultiPoint, Point, Polygon, box  # noqa: E402

try:
    import geopandas as gpd
except ImportError:                                  # pragma: no cover
    gpd = None

needs_geopandas = pytest.mark.skipif(gpd is None, reason="geopandas not installed")


def grid_boxes(side=4, shrink=0.0):
    """side × side unit squares; ``shrink`` opens a gap between them."""
    return [box(i + shrink, j + shrink, i + 1 - shrink, j + 1 - shrink)
            for i in range(side) for j in range(side)]


def assert_proper(colors, pairs):
    assert len(pairs), "expected a non-empty adjacency graph"
    assert (colors[pairs[:, 0]] != colors[pairs[:, 1]]).all()


# --------------------------------------------------------------- adjacency


def test_connect_finds_edge_sharing_neighbors_only():
    """A 4x4 grid has 24 rook adjacencies; the 18 corner touches don't count."""
    pairs = geo.connect(grid_boxes(4))
    assert pairs.shape == (24, 2)
    assert pairs.dtype == np.int32
    assert (pairs[:, 0] < pairs[:, 1]).all()         # (lo, hi), deduplicated


def test_point_contacts_count_when_the_filter_is_disabled():
    pairs = geo.connect(grid_boxes(4), min_shared_length=None)
    assert pairs.shape == (24 + 18, 2)


def test_min_shared_length_drops_short_seams():
    """A sliver of contact is filtered out by a length threshold."""
    a = box(0, 0, 1, 1)
    b = box(1, 0, 2, 0.05)                           # 0.05-long shared edge
    c = box(1, 0.5, 2, 1.5)                          # 0.5-long shared edge
    assert len(geo.connect([a, b, c])) == 2
    assert len(geo.connect([a, b, c], min_shared_length=0.1)) == 1


def test_tolerance_closes_hairline_gaps():
    gapped = grid_boxes(4, shrink=0.01)              # 0.02 between neighbors
    assert len(geo.connect(gapped)) == 0
    assert len(geo.connect(gapped, tolerance=0.05)) == 24 + 18


def test_min_shared_length_applies_across_a_tolerance_gap():
    """With a tolerance, contact is measured on the dilated geometries."""
    a = box(0, 0, 1, 1)
    b = box(1.02, 0, 2, 0.05)                        # near-miss sliver
    c = box(1.02, 0.5, 2, 1.5)                       # near-miss long seam
    assert len(geo.connect([a, b, c], tolerance=0.05)) == 2
    pairs = geo.connect([a, b, c], tolerance=0.05, min_shared_length=0.2)
    assert pairs.tolist() == [[0, 2]]


def test_rejected_pairs_are_returned_on_request():
    pairs, rejected = geo.connect(grid_boxes(4), return_rejected=True)
    assert len(pairs) == 24 and len(rejected) == 18


def test_non_areal_geometries_bypass_the_length_filter():
    """Lines meet at points; a zero-length contact is all they can have."""
    lines = [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])]
    assert len(geo.connect(lines)) == 1
    mixed = [box(0, 0, 1, 1), Point(0.5, 0.5)]
    assert len(geo.connect(mixed)) == 1


def test_missing_and_empty_geometries_take_part_in_no_pairs():
    geoms = [box(0, 0, 1, 1), None, Polygon(), box(1, 0, 2, 1)]
    assert geo.connect(geoms).tolist() == [[0, 3]]


def test_matches_the_raster_adjacency_for_the_same_layout():
    """Rectangles as pixels and as polygons must give the same graph."""
    img = np.zeros((8, 8), dtype=np.int32)
    polys = []
    for k, (r0, r1, c0, c1) in enumerate(
            [(0, 4, 0, 4), (0, 4, 4, 8), (4, 8, 0, 3), (4, 8, 3, 8)], start=1):
        img[r0:r1, c0:c1] = k
        polys.append(box(c0, r0, c1, r1))
    raster = {tuple(p) for p in (ncolor.connect(img, conn=1) - 1)}
    vector = {tuple(p) for p in geo.connect(polys)}
    assert raster == vector


# ---------------------------------------------------------------- coloring


def test_label_colors_a_planar_partition():
    polys = grid_boxes(6)
    colors, n = geo.label(polys, return_n=True)
    assert colors.dtype == np.uint8
    assert colors.shape == (36,)
    assert n <= 4 and colors.min() >= 1
    assert_proper(colors, geo.connect(polys))


def test_label_on_a_voronoi_tessellation():
    """The realistic case: a few thousand irregular, exactly-shared borders."""
    rng = np.random.default_rng(0)
    pts = MultiPoint(rng.random((400, 2)) * 100)
    cells = list(shapely.voronoi_polygons(pts, extend_to=box(0, 0, 100, 100)).geoms)
    colors, n, conflicts = geo.label(
        cells, return_n=True, return_conflicts=True)
    assert n <= 4 and conflicts == 0
    assert_proper(colors, geo.connect(cells))


def test_label_respects_a_tolerance():
    gapped = grid_boxes(4, shrink=0.01)
    assert set(geo.label(gapped).tolist()) == {1}   # no edges, all 1
    colors, n = geo.label(gapped, tolerance=0.05, return_n=True)
    assert n <= 4
    assert_proper(colors, geo.connect(gapped, tolerance=0.05))


def test_missing_geometries_get_color_zero():
    colors = geo.label([box(0, 0, 1, 1), None, Polygon(), box(1, 0, 2, 1)])
    assert colors[1] == 0 and colors[2] == 0
    assert colors[0] != colors[3] and colors[0] > 0


def test_soft_edges_separate_rejected_neighbors_when_possible():
    """Corner touches aren't hard constraints but should still differ."""
    polys = grid_boxes(4)
    hard, corners = geo.connect(polys, return_rejected=True)

    on = geo.label(polys, soft=True)
    off = geo.label(polys, soft=False)
    for colors in (on, off):
        assert_proper(colors, hard)                  # hard graph always holds
    shared = lambda c: int((c[corners[:, 0]] == c[corners[:, 1]]).sum())
    assert shared(on) == 0
    assert shared(off) > 0                           # unconstrained: some clash


def test_soft_edges_with_no_hard_edges_terminate():
    """Regression: a 1-color palette used to hang the soft search."""
    polys = [box(0, 0, 1, 1), box(1, 1, 2, 2)]       # one point of contact
    assert len(geo.connect(polys)) == 0
    colors, n = geo.label(polys, return_n=True)
    assert n == 1 and colors.tolist() == [1, 1]


def test_empty_and_single_inputs():
    assert geo.label([]).shape == (0,)
    assert geo.label(box(0, 0, 1, 1)).tolist() == [1]


def test_n_budget_is_honored():
    """A K_5 of mutually-overlapping polygons needs five colors."""
    overlapping = [box(0, 0, 10, 10).buffer(0),
                   box(1, 1, 11, 11), box(2, 2, 12, 12),
                   box(3, 3, 13, 13), box(4, 4, 14, 14)]
    colors, n = geo.label(overlapping, return_n=True)
    assert n == 5                                    # escalated past the budget
    assert len(set(colors.tolist())) == 5


# ------------------------------------------------------------ input formats


def test_accepts_a_sequence_of_shapely_geometries():
    assert geo.label(grid_boxes(3)).shape == (9,)


def test_accepts_objects_with_geo_interface():
    class Feature:
        def __init__(self, geom):
            self.__geo_interface__ = geom.__geo_interface__

    feats = [Feature(g) for g in grid_boxes(3)]
    assert_proper(geo.label(feats), geo.connect(feats))


def test_accepts_a_geojson_mapping():
    polys = grid_boxes(3)
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {},
                        "geometry": g.__geo_interface__} for g in polys]}
    assert_proper(geo.label(fc), geo.connect(polys))
    assert geo.label(json.dumps(fc)).shape == (9,)
    # A bare geometry mapping and a GeometryCollection are accepted too.
    assert geo.label(polys[0].__geo_interface__).shape == (1,)
    gc = {"type": "GeometryCollection",
          "geometries": [g.__geo_interface__ for g in polys]}
    assert geo.label(gc).shape == (9,)


def test_geojson_feature_with_null_geometry():
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {}, "geometry": None},
                       {"type": "Feature", "properties": {},
                        "geometry": box(0, 0, 1, 1).__geo_interface__}]}
    assert geo.label(fc).tolist() == [0, 1]


def test_mapping_without_a_type_is_rejected():
    with pytest.raises(ValueError, match="GeoJSON"):
        geo.label({"not": "geojson"})


def test_unsupported_element_type_is_rejected():
    with pytest.raises(TypeError, match="geometry"):
        geo.label([box(0, 0, 1, 1), 42])


@needs_geopandas
def test_accepts_a_geodataframe_and_geoseries():
    polys = grid_boxes(4)
    gdf = gpd.GeoDataFrame({"name": list(range(16))}, geometry=polys,
                           crs="EPSG:3857")
    colors = geo.label(gdf)
    assert_proper(colors, geo.connect(gdf))
    assert np.array_equal(colors, geo.label(gdf.geometry))


@needs_geopandas
def test_return_frame_keeps_the_other_columns_and_crs():
    gdf = gpd.GeoDataFrame({"name": list(range(9))}, geometry=grid_boxes(3),
                           crs="EPSG:3857")
    out = geo.label(gdf, return_frame=True, column="ncolor")
    assert isinstance(out, gpd.GeoDataFrame)
    assert list(out.columns) == ["name", "geometry", "ncolor"]
    assert out.crs == gdf.crs
    assert "ncolor" not in gdf.columns               # input left alone

    from_list = geo.label(grid_boxes(3), return_frame=True)
    assert isinstance(from_list, gpd.GeoDataFrame)
    assert "color" in from_list.columns


@needs_geopandas
def test_accepts_a_geojson_file_path(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=grid_boxes(3), crs="EPSG:4326")
    path = tmp_path / "cells.geojson"
    gdf.to_file(path, driver="GeoJSON")
    assert geo.label(str(path)).shape == (9,)
    assert geo.label(os.fspath(path)).shape == (9,)


def test_raster_label_points_at_geo_label_for_vector_input():
    with pytest.raises(TypeError, match=r"geo\.label"):
        ncolor.label(box(0, 0, 1, 1))


@needs_geopandas
def test_raster_label_points_at_geo_label_for_a_geodataframe():
    gdf = gpd.GeoDataFrame(geometry=grid_boxes(2))
    with pytest.raises(TypeError, match=r"geo\.label"):
        ncolor.label(gdf)


@needs_geopandas
def test_return_frame_from_a_geoseries_keeps_the_crs_and_index():
    """A GeoSeries has no .assign, so the frame is rebuilt; carry both over."""
    gs = gpd.GeoSeries(grid_boxes(3), crs="EPSG:3857",
                       index=list(range(100, 109)))
    out = geo.label(gs, return_frame=True)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == gs.crs
    assert out.index.tolist() == gs.index.tolist()
    assert "color" in out.columns


@needs_geopandas
def test_return_frame_from_a_bare_geometry_list_has_no_crs():
    out = geo.label(grid_boxes(3), return_frame=True)
    assert isinstance(out, gpd.GeoDataFrame) and out.crs is None


@needs_geopandas
def test_dataframe_without_an_active_geometry_column_says_so():
    bare = gpd.GeoDataFrame({"v": range(4), "shape": grid_boxes(2)})
    with pytest.raises(TypeError, match="geometry column"):
        geo.label(bare)
    import pandas as pd
    with pytest.raises(TypeError, match="geometry column"):
        geo.label(pd.DataFrame({"a": [1, 2]}))


@needs_geopandas
def test_return_frame_replaces_an_existing_column_without_touching_the_input():
    gdf = gpd.GeoDataFrame({"color": ["red"] * 9}, geometry=grid_boxes(3),
                           crs="EPSG:3857")
    out = geo.label(gdf, return_frame=True)
    assert out["color"].tolist() != ["red"] * 9      # replaced in the copy
    assert gdf["color"].tolist() == ["red"] * 9      # input untouched


@needs_geopandas
def test_geodataframe_with_a_renamed_geometry_column_and_odd_index():
    gdf = gpd.GeoDataFrame({"v": [10, 20, 30]},
                           geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1),
                                     box(2, 0, 3, 1)],
                           index=[7, 3, 99], crs="EPSG:4326")
    gdf = gdf.rename_geometry("shape")
    out = geo.label(gdf, return_frame=True)
    assert out.index.tolist() == [7, 3, 99]          # positional, not by label
    assert out["color"].iloc[0] != out["color"].iloc[1]


def test_geojson_file_without_geopandas(tmp_path, monkeypatch):
    """The reader falls back to plain json when GeoPandas is absent."""
    import sys
    polys = grid_boxes(3)
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {},
                        "geometry": g.__geo_interface__} for g in polys]}
    path = tmp_path / "plain.geojson"
    path.write_text(json.dumps(fc))
    monkeypatch.setitem(sys.modules, "geopandas", None)   # import -> ImportError
    colors = geo.label(str(path))
    assert colors.shape == (9,)
    assert_proper(colors, geo.connect(str(path)))
