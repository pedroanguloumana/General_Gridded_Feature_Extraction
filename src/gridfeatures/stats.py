"""Built-in statistic functions.

Every function takes a single :class:`~gridfeatures.feature.Feature` and returns
a number. Use them directly in a Config's ``statistics`` dict, mix them with
your own, or use the parameterized factories (``core_size``) to bind a
threshold.

Example
-------
>>> from gridfeatures import stats
>>> statistics = {
...     "size_px": stats.size,
...     "max_precip": stats.max,
...     "swath_edge_px": stats.swath_edge_pixels,
...     "core_10mm_px": stats.core_size(10.0),
...     "total_precip": stats.total,
... }
"""

import numpy as np
from scipy import ndimage

from .detection import _get_comparator

# 4-connectivity neighbour offsets (row, col).
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --- simple reductions over member cells ---------------------------------
def size(f):
    """Total number of pixels/cells in the feature."""
    return f.size


def maximum(f):
    """Maximum field value within the feature (e.g. peak precip rate)."""
    return float(np.nanmax(f.values))


def minimum(f):
    """Minimum field value within the feature."""
    return float(np.nanmin(f.values))


def mean(f):
    """Unweighted mean field value within the feature."""
    return float(np.nanmean(f.values))


def total(f):
    """Area-weighted sum of the field (e.g. total precip volume, mm*km^2)."""
    return float(np.nansum(f.values * f.area))


def area_km2(f):
    """Total feature area in km^2."""
    return float(np.nansum(f.area))


def centroid_lat(f):
    """Area-weighted centroid latitude (deg north)."""
    return f.centroid[0]


def centroid_lon(f):
    """Area-weighted centroid longitude (deg east)."""
    return f.centroid[1]


# Convenient aliases matching common naming (shadow builtins only in this module).
max = maximum
min = minimum


# --- boundary / swath edge ------------------------------------------------
def touches_boundary(f):
    """True if the feature touches the grid edge or a NaN (missing-data) cell.

    This flags features that may extend beyond the observable domain - the
    generic analogue of a real GPM swath edge.
    """
    ny, nx = f.grid_shape
    if (
        (f.rows == 0).any()
        or (f.rows == ny - 1).any()
        or (f.cols == 0).any()
        or (f.cols == nx - 1).any()
    ):
        return True
    # Not on the grid edge, so all 4-neighbours are in range; check for NaNs.
    field = f._ctx.field
    for dr, dc in _NEIGHBORS:
        if np.isnan(field[f.rows + dr, f.cols + dc]).any():
            return True
    return False


def swath_edge_pixels(f):
    """Number of feature pixels bordering an artificial-swath edge.

    A pixel counts if any of its 4-neighbours lies in a *different* swath
    strip. Requires ``use_swath=True`` in the Config (raises otherwise). This
    is the metric a reviewer asked for: how much of a feature sits against a
    swath seam and would be clipped by a real swath.
    """
    swath = f._ctx.swath
    if swath is None:
        raise ValueError(
            "swath_edge_pixels requires use_swath=True in the Config."
        )
    ny, nx = swath.shape
    rows, cols = f.rows, f.cols
    own = swath[rows, cols]
    is_edge = np.zeros(rows.shape, dtype=bool)
    for dr, dc in _NEIGHBORS:
        rr = rows + dr
        cc = cols + dc
        valid = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        diff = np.zeros(rows.shape, dtype=bool)
        diff[valid] = swath[rr[valid], cc[valid]] != own[valid]
        is_edge |= diff
    return int(is_edge.sum())


# --- parameterized factories ---------------------------------------------
def core_size(core_threshold, comparison=">=", connectivity=None):
    """Factory: size (pixels) of the largest contiguous *core* within a feature.

    A core is the set of feature cells whose values satisfy
    ``value <comparison> core_threshold``. The returned function reports the
    pixel count of the single largest contiguous such region. Example: within
    features defined by a 1 mm/hr threshold, ``core_size(10.0)`` gives the size
    of the biggest >=10 mm/hr core.

    Parameters
    ----------
    core_threshold : float
        The higher (core) threshold.
    comparison : str
        Comparison operator (default ``>=``).
    connectivity : int or None
        Connectivity for the core labeling; defaults to the detection
        connectivity used for the parent feature.
    """
    op = _get_comparator(comparison)

    def _core_size(f):
        conn = connectivity if connectivity is not None else f._ctx.connectivity
        sub_field = f.local_field()
        core = f.local_mask() & op(sub_field, core_threshold) & ~np.isnan(sub_field)
        if not core.any():
            return 0
        structure = ndimage.generate_binary_structure(2, conn)
        labeled, n = ndimage.label(core, structure=structure)
        if n == 0:
            return 0
        counts = np.bincount(labeled.ravel())[1:]  # drop background bin
        return int(counts.max())

    _core_size.__name__ = f"core_size_{core_threshold}"
    _core_size.__doc__ = (
        f"Largest contiguous sub-region with value {comparison} {core_threshold} (pixels)."
    )
    return _core_size
