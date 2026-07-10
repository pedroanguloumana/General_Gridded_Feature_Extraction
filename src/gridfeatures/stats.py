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
def _boundary_mask(f, count_grid_edge):
    """Boolean array over the feature's member cells: is each one on the boundary?

    A cell is on the boundary if a 4-neighbour is a NaN (missing-data) cell, or
    - when ``count_grid_edge`` - if a 4-neighbour lies off the grid.
    """
    field = f._ctx.field
    ny, nx = field.shape
    rows, cols = f.rows, f.cols
    is_edge = np.zeros(rows.shape, dtype=bool)
    for dr, dc in _NEIGHBORS:
        rr = rows + dr
        cc = cols + dc
        off = (rr < 0) | (rr >= ny) | (cc < 0) | (cc >= nx)
        if count_grid_edge:
            is_edge |= off
        valid = ~off
        if valid.any():
            nan_neighbor = np.zeros(rows.shape, dtype=bool)
            nan_neighbor[valid] = np.isnan(field[rr[valid], cc[valid]])
            is_edge |= nan_neighbor
    return is_edge


def boundary_pixels_where(count_grid_edge=True):
    """Factory: count feature pixels bordering the observable-domain boundary.

    NaN 4-neighbours always count. Off-grid 4-neighbours count only when
    ``count_grid_edge`` is True.

    Parameters
    ----------
    count_grid_edge : bool
        Whether a pixel on the edge of the array counts as a boundary pixel.

        Use ``True`` (the default, and what :func:`boundary_pixels` does) for
        fields whose array edge really is a data boundary - a model domain
        edge, a regional cutout you intend to treat as closed.

        Use ``False`` for satellite swath crops such as GPM L2. There the array
        edges are the *along-track* cut (the granule time range) or a regional
        box clip, and the instrument observed straight through them; only NaN
        cells inside the grid mark the real *cross-track* swath edge. Counting
        grid edges there inflates the result by the whole along-track cap.

    Notes
    -----
    With ``count_grid_edge=False`` the count is slightly conservative: where a
    cross-track edge exits through a corner of a tight crop, the unobserved
    neighbour is off-grid rather than NaN, so a few pixels per swath are missed.
    Resolving that would need native along/cross scan coordinates.
    """
    def _boundary_pixels(f):
        return int(_boundary_mask(f, count_grid_edge).sum())

    suffix = "" if count_grid_edge else "_no_grid_edge"
    _boundary_pixels.__name__ = f"boundary_pixels{suffix}"
    _boundary_pixels.__doc__ = (
        "Number of feature pixels bordering a NaN cell"
        + (" or the grid edge." if count_grid_edge else " (grid edges ignored).")
    )
    return _boundary_pixels


def touches_boundary_where(count_grid_edge=True):
    """Factory: True if the feature has any boundary pixel.

    Boolean companion to :func:`boundary_pixels_where`; see it for the meaning
    of ``count_grid_edge``.
    """
    def _touches_boundary(f):
        return bool(_boundary_mask(f, count_grid_edge).any())

    suffix = "" if count_grid_edge else "_no_grid_edge"
    _touches_boundary.__name__ = f"touches_boundary{suffix}"
    return _touches_boundary


#: Number of feature pixels bordering the grid edge or a NaN cell.
#:
#: For a closed domain this is the count of pixels on the observable boundary.
#: For GPM L2 swath crops prefer :data:`cross_track_edge_pixels`, which does not
#: mistake the along-track cut for a swath edge.
boundary_pixels = boundary_pixels_where(count_grid_edge=True)

#: True if the feature touches the grid edge or a NaN (missing-data) cell.
touches_boundary = touches_boundary_where(count_grid_edge=True)

#: Number of feature pixels bordering a NaN cell, ignoring the grid edge.
#:
#: The right "swath edge" metric for real satellite swath data (e.g. GPM L2),
#: where the array edges are the along-track cut rather than the instrument's
#: cross-track edge.
cross_track_edge_pixels = boundary_pixels_where(count_grid_edge=False)

#: True if the feature borders a NaN cell, ignoring the grid edge.
touches_cross_track_edge = touches_boundary_where(count_grid_edge=False)


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
