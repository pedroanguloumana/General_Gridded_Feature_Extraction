"""Tests for spatial-extent and ellipse-fit (elongation) statistics."""

import numpy as np
import pytest

from gridfeatures import stats
from gridfeatures.config import Config
from gridfeatures.runner import extract_features


def _features(field, lats, lons, min_size=1):
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    config = Config(
        files="mem",
        variable="x",
        threshold=1.0,
        statistics={"size": stats.size},
        min_size=min_size,
    )
    return extract_features(field, lats2d, lons2d, config, source="mem")


def _one(field, lats, lons):
    feats = _features(field, lats, lons)
    assert len(feats) == 1
    return feats[0]


# --- cardinal extents -----------------------------------------------------
def test_extents_match_member_coords(simple_field):
    field, lats, lons = simple_field
    for f in _features(field, lats, lons):
        assert stats.north_extent(f) == float(f.lats.max())
        assert stats.south_extent(f) == float(f.lats.min())
        assert stats.east_extent(f) == float(f.lons.max())
        assert stats.west_extent(f) == float(f.lons.min())
        assert stats.lat_extent(f) == stats.north_extent(f) - stats.south_extent(f)
        assert stats.lon_extent(f) == stats.east_extent(f) - stats.west_extent(f)
        assert stats.lat_extent(f) >= 0.0
        assert stats.lon_extent(f) >= 0.0


def test_extents_known_values(simple_field):
    field, lats, lons = simple_field
    # Blob B occupies rows 4:6 (lats), cols 5:8 (lons).
    blob_b = next(f for f in _features(field, lats, lons) if f.size == 6)
    assert stats.south_extent(blob_b) == lats[4]
    assert stats.north_extent(blob_b) == lats[5]
    assert stats.west_extent(blob_b) == lons[5]
    assert stats.east_extent(blob_b) == lons[7]


# --- antimeridian handling ------------------------------------------------
def test_extents_straddling_antimeridian():
    # One feature (adjacent columns) whose longitudes cross +/-180:
    # 178 -> 179 -> 180(=-180) -> -179, i.e. a 3-degree-wide feature over the seam.
    field = np.full((2, 4), 5.0)
    lats = np.array([0.0, 1.0])
    lons = np.array([178.0, 179.0, -180.0, -179.0])
    f = _one(field, lats, lons)

    # True span is 3 deg, not the ~358 a naive max-min would give.
    assert stats.lon_extent(f) == pytest.approx(3.0)
    # West edge is the pre-seam side, east edge the post-seam side, so the
    # east extent is numerically *less* than the west extent -- the signature
    # of a feature that wraps past +180.
    assert stats.west_extent(f) == pytest.approx(178.0)
    assert stats.east_extent(f) == pytest.approx(-179.0)
    assert stats.east_extent(f) < stats.west_extent(f)
    # Latitude is unaffected by any of this.
    assert stats.lat_extent(f) == pytest.approx(1.0)


def test_lon_extent_invariant_to_longitude_origin():
    # The same feature shape, once over the seam and once far from it, must
    # report the same longitudinal span. Both grids are evenly spaced at 1 deg
    # once unrolled ([178,179,180,181] vs [10,11,12,13]), so the span is 3.
    field = np.full((2, 4), 5.0)
    lats = np.array([0.0, 1.0])
    seam = _one(field, lats, np.array([178.0, 179.0, -180.0, -179.0]))
    away = _one(field, lats, np.array([10.0, 11.0, 12.0, 13.0]))
    assert stats.lon_extent(seam) == pytest.approx(3.0)
    assert stats.lon_extent(seam) == pytest.approx(stats.lon_extent(away))


def test_unwrap_lons_no_straddle_is_identity():
    lons = np.array([10.0, 11.0, 12.0, 13.0])
    assert np.array_equal(stats._unwrap_lons(lons), lons)


def test_unwrap_lons_lifts_wrapped_side():
    lons = np.array([178.0, 179.0, -179.0, -178.0])
    out = stats._unwrap_lons(lons)
    # The two negative (post-seam) cells are lifted by 360 to sit past +180.
    assert np.array_equal(out, np.array([178.0, 179.0, 181.0, 182.0]))
    assert out.max() - out.min() == pytest.approx(4.0)


# --- eccentricity / elongation --------------------------------------------
def test_round_blob_low_eccentricity():
    # Symmetric 3x3 block centred on the equator: near-circular in km.
    field = np.full((3, 3), 5.0)
    lats = np.array([-1.0, 0.0, 1.0])
    lons = np.array([-1.0, 0.0, 1.0])
    f = _one(field, lats, lons)
    assert stats.eccentricity(f) == pytest.approx(0.0, abs=1e-9)
    assert stats.elongation(f) == pytest.approx(1.0, abs=1e-9)


def test_thin_line_is_maximally_eccentric():
    # A 1-cell-wide, 5-cell-long feature: zero minor axis.
    field = np.zeros((3, 5))
    field[1, :] = 5.0                      # single-row feature in a 3-row grid
    lats = np.array([-1.0, 0.0, 1.0])
    lons = np.linspace(0.0, 4.0, 5)
    f = _one(field, lats, lons)
    assert stats.eccentricity(f) == pytest.approx(1.0)
    assert stats.elongation(f) == np.inf
    assert stats.minor_axis_km(f) == pytest.approx(0.0)
    assert stats.major_axis_km(f) > 0.0


def test_rectangle_eccentricity_between():
    # 2x5 rectangle: elongated but not degenerate -> 0 < ecc < 1.
    field = np.full((2, 5), 5.0)
    lats = np.array([0.0, 1.0])
    lons = np.linspace(0.0, 4.0, 5)
    f = _one(field, lats, lons)
    ecc = stats.eccentricity(f)
    assert 0.0 < ecc < 1.0
    assert stats.elongation(f) > 1.0
    assert stats.major_axis_km(f) > stats.minor_axis_km(f) > 0.0


def test_single_pixel_is_circular():
    field = np.zeros((3, 3))
    field[1, 1] = 5.0
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    f = _one(field, lats, lons)
    assert stats.eccentricity(f) == 0.0
    assert stats.elongation(f) == pytest.approx(1.0)
    assert stats.major_axis_km(f) == pytest.approx(0.0)
    assert stats.minor_axis_km(f) == pytest.approx(0.0)


def test_shape_metrics_invariant_across_antimeridian():
    # Elongation is a shape property: a feature straddling the seam must fit the
    # same ellipse as the identical feature shifted away from it.
    field = np.full((2, 4), 5.0)
    lats = np.array([0.0, 1.0])
    seam = _one(field, lats, np.array([178.0, 179.0, -180.0, -179.0]))
    away = _one(field, lats, np.array([10.0, 11.0, 12.0, 13.0]))
    assert stats.eccentricity(seam) == pytest.approx(stats.eccentricity(away))
    assert stats.elongation(seam) == pytest.approx(stats.elongation(away))
    assert stats.major_axis_km(seam) == pytest.approx(stats.major_axis_km(away))
    assert stats.minor_axis_km(seam) == pytest.approx(stats.minor_axis_km(away))
    assert stats.orientation_deg(seam) == pytest.approx(stats.orientation_deg(away))


# --- orientation ----------------------------------------------------------
def test_orientation_zonal_vs_meridional():
    lats = np.array([0.0, 1.0, 2.0, 3.0])
    lons = np.array([0.0, 1.0, 2.0, 3.0])

    # East-west feature (one row, several columns) -> orientation ~ 0.
    ew = np.zeros((4, 4))
    ew[0, :] = 5.0
    f_ew = _one(ew, lats, lons)
    assert stats.orientation_deg(f_ew) == pytest.approx(0.0, abs=1e-6)

    # North-south feature (one column, several rows) -> orientation ~ 90.
    ns = np.zeros((4, 4))
    ns[:, 0] = 5.0
    f_ns = _one(ns, lats, lons)
    assert abs(stats.orientation_deg(f_ns)) == pytest.approx(90.0, abs=1e-6)


# --- coordinate-space switch ----------------------------------------------
def test_grid_space_factory_runs_and_bounded():
    field = np.full((2, 5), 5.0)
    lats = np.array([40.0, 41.0])       # off-equator: km grid is anisotropic
    lons = np.linspace(0.0, 4.0, 5)
    f = _one(field, lats, lons)
    ecc_grid = stats.ellipse_eccentricity(space="grid")(f)
    ecc_geo = stats.ellipse_eccentricity(space="geographic")(f)
    assert 0.0 <= ecc_grid <= 1.0
    assert 0.0 <= ecc_geo <= 1.0
    # Off the equator longitude cells are narrower in km, so this east-west
    # elongated footprint spans less ground E-W than its column count suggests
    # and reads as *less* elongated on the tangent plane than in index space.
    assert ecc_grid > ecc_geo


def test_bad_space_raises():
    field = np.full((2, 2), 5.0)
    f = _one(field, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        stats.ellipse_eccentricity(space="polar")(f)
