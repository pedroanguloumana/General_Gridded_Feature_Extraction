"""Dominant-strip swath statistics: px_in_swath and its matching edge count.

These tests hand-build the swath index array instead of deriving it from
``swath_index``, so the strips are exactly where the comments say they are and
the expected counts can be checked by hand. Strips are 3 columns wide
(``col // 3``), so seams sit between cols 2|3, 5|6, 8|9, ...
"""

import numpy as np

from gridfeatures import stats
from gridfeatures.config import Config
from gridfeatures.runner import extract_features, feature_row
from gridfeatures.swath import dominant_swath

STRIP_WIDTH = 3


def _strips(shape):
    """Vertical strips 3 columns wide: strip index = col // 3."""
    ny, nx = shape
    return np.tile(np.arange(nx) // STRIP_WIDTH, (ny, 1))


def _config(**kw):
    return Config(
        files="mem",
        variable="x",
        threshold=1.0,
        statistics={"size": stats.size},
        min_size=1,
        use_swath=True,
        **kw,
    )


def _one_feature(field):
    """Extract the single feature from ``field`` under strip-3 swath geometry."""
    ny, nx = field.shape
    lats = np.linspace(-4.0, 4.0, ny)
    lons = np.linspace(0.0, float(nx - 1), nx)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    feats = extract_features(
        field, lats2d, lons2d, _config(), source="mem", swath=_strips(field.shape)
    )
    assert len(feats) == 1
    return feats[0]


def _counts(f):
    """(numerator, denominator) of the dominant-strip edge fraction."""
    _, px_in_swath = dominant_swath(f.swath_index)
    return stats.swath_edge_pixels_in_dominant(f), px_in_swath


def _blank(shape):
    return np.zeros(shape, dtype=float)


# --- the four geometries -------------------------------------------------
def test_feature_inside_one_strip_has_no_edge():
    # Rows 0-2, cols 0-1: wholly inside strip 0, and pressed against the
    # domain's top-left corner. Col 1's right neighbour is col 2 -- still strip
    # 0 -- so nothing borders a seam, and the off-grid neighbours above and to
    # the left must not count (the count_grid_edge=False convention).
    field = _blank((8, 15))
    field[0:3, 0:2] = 5.0
    f = _one_feature(field)

    edge, px_in_swath = _counts(f)
    assert px_in_swath == 6
    assert edge == 0
    assert stats.swath_edge_fraction_in_dominant(f) == 0.0
    assert edge <= px_in_swath


def test_feature_straddling_a_seam_has_high_fraction():
    # Rows 2-4, cols 0-4: 9 px in strip 0 (cols 0,1,2), 6 px in strip 1 (cols
    # 3,4). Dominant strip 0; its col-2 pixels (3 of them) border the 2|3 seam.
    field = _blank((8, 15))
    field[2:5, 0:5] = 5.0
    f = _one_feature(field)

    edge, px_in_swath = _counts(f)
    assert dominant_swath(f.swath_index)[0] == 0
    assert px_in_swath == 9
    assert edge == 3
    assert stats.swath_edge_fraction_in_dominant(f) == 3 / 9   # 33%, rejected
    assert edge <= px_in_swath


def test_feature_barely_crossing_a_seam_has_low_fraction():
    # A 10x2 block deep inside strip 0 (20 px), plus a 1-px arm reaching across
    # the 2|3 seam: (5,2) in strip 0, (5,3) in strip 1. Only the arm's own
    # strip-0 pixel borders the seam -> 1/21 = 4.8%, kept under a 5% threshold.
    field = _blank((10, 15))
    field[0:10, 0:2] = 5.0
    field[5, 2] = 5.0
    field[5, 3] = 5.0
    f = _one_feature(field)

    edge, px_in_swath = _counts(f)
    assert dominant_swath(f.swath_index)[0] == 0
    assert px_in_swath == 21
    assert edge == 1
    assert stats.swath_edge_fraction_in_dominant(f) < 0.05
    assert edge <= px_in_swath


def test_three_strips_with_a_tie_for_the_majority():
    # Rows 3-4, cols 3-8, plus a single pixel at (3,2). Strip counts are
    # {0: 1, 1: 6, 2: 6}: three strips, and strips 1 and 2 tie for the
    # plurality. np.unique's sorted order + argmax gives the tie to strip 1.
    field = _blank((8, 15))
    field[3:5, 3:9] = 5.0
    field[3, 2] = 5.0
    f = _one_feature(field)

    values, counts = np.unique(f.swath_index, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {0: 1, 1: 6, 2: 6}

    swath_id, px_in_swath = dominant_swath(f.swath_index)
    assert swath_id == 1          # lowest index wins the tie
    assert px_in_swath == 6

    # Strip 1's pixels at col 3 border the 2|3 seam and those at col 5 border
    # the 5|6 seam: 2 + 2 = 4 of its 6 pixels.
    edge = stats.swath_edge_pixels_in_dominant(f)
    assert edge == 4
    assert edge <= px_in_swath


# --- the invariant the whole design exists to protect --------------------
def test_edge_count_never_exceeds_px_in_swath():
    fields = []
    for geometry in ("block", "straddle", "arm", "tie"):
        field = _blank((10, 15))
        if geometry == "block":
            field[0:3, 0:2] = 5.0
        elif geometry == "straddle":
            field[2:5, 0:5] = 5.0
        elif geometry == "arm":
            field[0:10, 0:2] = 5.0
            field[5, 2:4] = 5.0
        else:
            field[3:5, 3:9] = 5.0
            field[3, 2] = 5.0
        fields.append(field)

    for field in fields:
        f = _one_feature(field)
        edge, px_in_swath = _counts(f)
        assert 0 <= edge <= px_in_swath
        assert 0.0 <= stats.swath_edge_fraction_in_dominant(f) <= 1.0


def test_whole_feature_swath_edge_pixels_is_not_a_valid_numerator():
    # Why the new stat exists. swath_edge_pixels tests each pixel against its
    # own strip, so on the 3-strip feature it counts seam pixels in every strip
    # -- 9 of them, more than the dominant strip's 6 pixels. Using it over
    # px_in_swath would give a fraction of 1.5.
    field = _blank((8, 15))
    field[3:5, 3:9] = 5.0
    field[3, 2] = 5.0
    f = _one_feature(field)

    _, px_in_swath = dominant_swath(f.swath_index)
    assert stats.swath_edge_pixels(f) == 9
    assert stats.swath_edge_pixels(f) > px_in_swath
    assert stats.swath_edge_pixels_in_dominant(f) <= px_in_swath


def test_requires_use_swath():
    field = _blank((8, 15))
    field[2:5, 0:5] = 5.0
    lats = np.linspace(-4.0, 4.0, 8)
    lons = np.linspace(0.0, 14.0, 15)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    config = Config(
        files="mem", variable="x", threshold=1.0, statistics={"size": stats.size}
    )
    f = extract_features(field, lats2d, lons2d, config, source="mem", swath=None)[0]

    for fn in (stats.swath_edge_pixels_in_dominant, stats.swath_edge_fraction_in_dominant):
        try:
            fn(f)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} should raise without a swath")


# --- runner column -------------------------------------------------------
def test_px_in_swath_column_matches_the_labelled_strip():
    # px_in_swath must be the pixel count of the strip named by swath_id, and
    # the edge stat must be a subset of it -- the consistency the shared helper
    # buys us.
    field = _blank((8, 15))
    field[3:5, 3:9] = 5.0
    field[3, 2] = 5.0
    f = _one_feature(field)

    config = _config()
    config.statistics["swath_edge_px_dom"] = stats.swath_edge_pixels_in_dominant
    row = feature_row(f, config)

    assert row["swath_id"] == 1
    assert row["px_in_swath"] == 6
    assert row["n_swaths"] == 3
    assert row["crosses_swath_boundary"] is True
    # px_in_swath is the count of swath_id's strip specifically, not just the
    # largest count under some other rule.
    assert row["px_in_swath"] == int((f.swath_index == row["swath_id"]).sum())
    assert row["swath_edge_px_dom"] <= row["px_in_swath"]


def test_px_in_swath_equals_size_for_a_single_strip_feature():
    # A feature inside one strip is entirely "within the swath", so the DYAMOND
    # denominator collapses to the GPM one (stats.size).
    field = _blank((8, 15))
    field[0:3, 0:2] = 5.0
    f = _one_feature(field)
    row = feature_row(f, _config())

    assert row["px_in_swath"] == f.size == 6
    assert row["n_swaths"] == 1
    assert row["crosses_swath_boundary"] is False


def test_px_in_swath_absent_when_swath_disabled():
    field = _blank((8, 15))
    field[2:5, 0:5] = 5.0
    lats = np.linspace(-4.0, 4.0, 8)
    lons = np.linspace(0.0, 14.0, 15)
    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    config = Config(
        files="mem", variable="x", threshold=1.0, statistics={"size": stats.size}
    )
    f = extract_features(field, lats2d, lons2d, config, source="mem", swath=None)[0]
    row = feature_row(f, config)

    assert "px_in_swath" not in row
    assert "swath_id" not in row
