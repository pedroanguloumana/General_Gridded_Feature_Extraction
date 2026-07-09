import numpy as np

from gridfeatures.swath import swath_index


def test_swath_indices_start_at_zero_and_increase():
    lats = np.linspace(0.0, 10.0, 11)
    lons = np.linspace(0.0, 10.0, 11)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    idx = swath_index(lats2d, lons2d, width_km=200.0, angle_deg=0.0)
    assert idx.min() == 0
    assert idx.max() >= 1  # domain wider than one 200 km strip


def test_angle_zero_strips_are_east_west():
    # angle 0 -> strips separated in the north (latitude) direction only.
    lats = np.linspace(0.0, 10.0, 11)
    lons = np.linspace(0.0, 10.0, 11)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    idx = swath_index(lats2d, lons2d, width_km=200.0, angle_deg=0.0)
    # Every row (fixed lat) should share one swath index across all lons.
    for row in idx:
        assert len(np.unique(row)) == 1


def test_angle_ninety_strips_are_north_south():
    lats = np.linspace(0.0, 10.0, 11)
    lons = np.linspace(0.0, 10.0, 11)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    idx = swath_index(lats2d, lons2d, width_km=200.0, angle_deg=90.0)
    # Every column (fixed lon) should share one swath index across all lats.
    for col in idx.T:
        assert len(np.unique(col)) == 1


def test_positive_width_required():
    lats2d, lons2d = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    try:
        swath_index(lats2d, lons2d, width_km=0.0, angle_deg=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive width")
