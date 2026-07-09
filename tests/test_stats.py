import numpy as np

from gridfeatures import stats
from gridfeatures.grid import cell_area_km2
from gridfeatures.runner import extract_features
from gridfeatures.config import Config
from gridfeatures.swath import swath_index


def _features(field, lats, lons, min_size=1, use_swath=False):
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    config = Config(
        files="mem",
        variable="x",
        threshold=1.0,
        statistics={"size": stats.size},
        min_size=min_size,
    )
    swath = None
    if use_swath:
        swath = swath_index(lats2d, lons2d, width_km=200.0, angle_deg=0.0)
    return extract_features(
        field, lats2d, lons2d, config, source="mem", swath=swath
    )


def test_size_and_max(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons)
    by_size = {f.size: f for f in feats}
    blob_a = by_size[4]  # 2x2
    assert stats.size(blob_a) == 4
    assert stats.max(blob_a) == 12.0


def test_min_size_filter(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons, min_size=2)
    # Tiny 1-cell blob is dropped -> 2 features remain.
    assert len(feats) == 2
    assert all(f.size >= 2 for f in feats)


def test_core_size(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons)
    blob_a = next(f for f in feats if f.size == 4)
    blob_b = next(f for f in feats if f.size == 6)
    core10 = stats.core_size(10.0)
    assert core10(blob_a) == 1   # only the single 12.0 cell exceeds 10
    assert core10(blob_b) == 0   # blob B maxes at 3.0


def test_core_size_largest_contiguous():
    # One feature (all > 1) with two separate cores >= 10.
    field = np.array(
        [
            [11.0, 11.0, 2.0, 12.0],
            [11.0, 2.0, 2.0, 2.0],
        ]
    )
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0, 2.0, 3.0])
    feats = _features(field, lats, lons)
    assert len(feats) == 1
    # Largest core is the 3-cell L-shape, not the isolated single cell.
    assert stats.core_size(10.0)(feats[0]) == 3


def test_total_is_area_weighted(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons)
    blob_b = next(f for f in feats if f.size == 6)
    area = cell_area_km2(lats, lons)
    # total == sum(value * area) over member cells.
    expected = float((blob_b.values * blob_b.area).sum())
    assert np.isclose(stats.total(blob_b), expected)
    assert stats.total(blob_b) > 0


def test_swath_edge_pixels_requires_swath(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons, use_swath=False)
    try:
        stats.swath_edge_pixels(feats[0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without swath")


def test_swath_edge_pixels_counts_seam():
    # A 2x6 feature crossed by vertical swath seams.
    field = np.ones((2, 6)) * 5.0
    lats = np.array([0.0, 1.0])
    lons = np.linspace(0.0, 50.0, 6)  # ~10 deg spacing -> wide in km
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    # angle 90 -> strips separate east-west, so seams cut across the wide
    # longitude span of this feature. Narrow width -> several seams.
    swath = swath_index(lats2d, lons2d, width_km=500.0, angle_deg=90.0)
    config = Config(
        files="mem", variable="x", threshold=1.0,
        statistics={"s": stats.size},
    )
    feats = extract_features(
        field, lats2d, lons2d, config, source="mem", swath=swath
    )
    assert len(feats) == 1
    n_edge = stats.swath_edge_pixels(feats[0])
    # With multiple swaths present, at least some cells border a seam.
    assert n_edge >= 1
    assert n_edge <= feats[0].size


def test_touches_boundary(simple_field):
    field, lats, lons = simple_field
    feats = _features(field, lats, lons)
    # Blob B occupies the last two columns -> touches the grid edge.
    blob_b = next(f for f in feats if f.size == 6)
    assert stats.touches_boundary(blob_b) is True
    # Blob A is interior.
    blob_a = next(f for f in feats if f.size == 4)
    assert stats.touches_boundary(blob_a) is False
