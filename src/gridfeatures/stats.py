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

import math

import numpy as np
from scipy import ndimage

from .detection import _get_comparator
from .swath import dominant_swath

# 4-connectivity neighbour offsets (row, col).
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Kilometres per degree of latitude (great-circle, mean Earth radius 6371 km).
# Also used for longitude after scaling by cos(latitude).
_KM_PER_DEG = math.pi * 6371.0 / 180.0


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
# NOTE: these shadow the builtins `max`/`min` for the rest of this module, so
# code below uses `np.maximum`/explicit comparisons rather than bare max()/min().
max = maximum
min = minimum


# --- longitude / antimeridian handling ------------------------------------
def _wrap180(lon):
    """Wrap a longitude (scalar or array) into the canonical [-180, 180) range."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def _unwrap_lons(lons):
    """Unroll longitudes into a contiguous arc, healing the antimeridian seam.

    Member cells lie on an arc of the longitude circle. This finds the *largest
    empty gap* between neighbouring cell longitudes and cuts the circle there,
    then lifts the cells east of the cut by 360 deg so the whole feature becomes
    contiguous and monotone-friendly. A feature straddling +/-180 thus reads as
    ``[179, 180, 181]`` instead of ``[179, -179]``; a feature nowhere near the
    seam is returned unchanged.

    Returned values may exceed +180 (that is the point); wrap back with
    :func:`_wrap180` when you need canonical longitudes. The heuristic assumes
    the feature spans well under 180 deg of longitude, which holds for any real
    contiguous feature - the seam is unambiguous only if there *is* a clear gap.
    """
    lons = np.asarray(lons, dtype=float)
    if lons.size <= 1:
        return lons.copy()
    s = np.sort(lons)
    gaps = np.diff(s)
    wrap_gap = (s[0] + 360.0) - s[-1]
    if gaps.size == 0 or wrap_gap >= gaps.max():
        return lons.copy()  # largest gap already spans the seam: no straddle
    cut_lo = s[int(np.argmax(gaps))]  # cells at or below this are the wrapped side
    out = lons.copy()
    out[out <= cut_lo] += 360.0
    return out


# --- spatial extent -------------------------------------------------------
# The four cardinal extents are the bounding coordinates of the feature's
# member cells. Longitude is handled across the +/-180 antimeridian via
# `_unwrap_lons`: a feature straddling the seam reports its true (narrow) span,
# with `east_extent` numerically less than `west_extent` - the signature of a
# feature that wraps past +180. This healing assumes the feature spans well
# under 180 deg of longitude (true for any real contiguous feature).
def north_extent(f):
    """Northernmost latitude of the feature (deg north)."""
    return float(np.nanmax(f.lats))


def south_extent(f):
    """Southernmost latitude of the feature (deg north)."""
    return float(np.nanmin(f.lats))


def east_extent(f):
    """Easternmost longitude of the feature (deg east), antimeridian-aware.

    For a feature crossing +/-180 this is the eastern (post-seam) edge and will
    be numerically *less* than :func:`west_extent`.
    """
    return float(_wrap180(_unwrap_lons(f.lons).max()))


def west_extent(f):
    """Westernmost longitude of the feature (deg east), antimeridian-aware."""
    return float(_wrap180(_unwrap_lons(f.lons).min()))


def lat_extent(f):
    """Latitudinal span (deg): ``north_extent - south_extent``."""
    return north_extent(f) - south_extent(f)


def lon_extent(f):
    """Longitudinal span (deg), antimeridian-aware.

    The angular width of the feature's longitude arc. Always non-negative and
    correct across the +/-180 seam (where a naive ``east - west`` would report
    ~360 for a narrow feature).
    """
    u = _unwrap_lons(f.lons)
    return float(u.max() - u.min())


# --- ellipse fit / elongation ---------------------------------------------
def _ellipse_moments(f, space):
    """Second-moment (inertia-tensor) eigen-analysis of the feature footprint.

    Fits the equivalent ellipse of the feature's *shape* - every member cell
    counts once, unweighted by value or area, matching the usual image-moments
    convention (e.g. scikit-image ``regionprops``).

    Parameters
    ----------
    space : {"geographic", "grid"}
        Coordinate system the moments are computed in.

        - ``"geographic"`` projects the member cells onto a local east/north
          tangent plane in kilometres (longitude scaled by ``cos(mean lat)``),
          so elongation reflects true ground distance. This is what you want
          for physical statements about feature shape, since grid cells are not
          square in km away from the equator. Longitudes are unrolled across the
          antimeridian first (:func:`_unwrap_lons`), so straddling features fit
          correctly.
        - ``"grid"`` uses raw (row, col) pixel indices - dimensionless, matches
          a plain ``regionprops`` call, but anisotropic in km off the equator.
          (Grid indices are already seam-free.)

    Returns
    -------
    (l1, l2, theta) : tuple of float
        ``l1 >= l2 >= 0`` are the eigenvalues of the coordinate covariance
        (each a variance, km^2 for geographic), and ``theta`` is the major-axis
        orientation in radians, counterclockwise from the x-axis (east, or
        increasing column).
    """
    if space == "geographic":
        lats = np.asarray(f.lats, dtype=float)
        lons = _unwrap_lons(f.lons)
        lat0 = float(lats.mean())
        y = (lats - lat0) * _KM_PER_DEG
        x = (lons - float(lons.mean())) * _KM_PER_DEG * math.cos(math.radians(lat0))
    elif space == "grid":
        y = f.rows.astype(float)
        x = f.cols.astype(float)
    else:
        raise ValueError(f"space must be 'geographic' or 'grid', got {space!r}")

    n = y.size
    y = y - y.mean()
    x = x - x.mean()
    cyy = float((y * y).sum()) / n
    cxx = float((x * x).sum()) / n
    cxy = float((x * y).sum()) / n

    tr = cxx + cyy
    # Discriminant is a sum of squares, so it is always non-negative.
    root = math.sqrt(((cxx - cyy) / 2.0) ** 2 + cxy * cxy)
    l1 = tr / 2.0 + root
    l2 = tr / 2.0 - root
    if l2 < 0.0:  # tiny negative from round-off
        l2 = 0.0
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    return l1, l2, theta


def ellipse_eccentricity(space="geographic"):
    """Factory: eccentricity of the feature's equivalent ellipse, in [0, 1].

    ``sqrt(1 - (minor/major)^2)`` where the axis lengths come from the fitted
    ellipse (:func:`_ellipse_moments`). 0 is a circle (or a single pixel); it
    approaches 1 as the feature becomes a thin line. Use this to rank features
    by how elongated they are, independent of size.

    ``space`` selects the coordinate system - see :func:`_ellipse_moments`.
    The module-level :data:`eccentricity` is this with ``space="geographic"``.
    """
    def _eccentricity(f):
        l1, l2, _ = _ellipse_moments(f, space)
        if l1 <= 0.0:
            return 0.0
        val = 1.0 - l2 / l1
        return float(math.sqrt(val)) if val > 0.0 else 0.0

    _eccentricity.__name__ = f"eccentricity_{space}"
    _eccentricity.__doc__ = (
        f"Equivalent-ellipse eccentricity in [0, 1] ({space} coordinates)."
    )
    return _eccentricity


def ellipse_elongation(space="geographic"):
    """Factory: aspect ratio of the feature's equivalent ellipse, >= 1.

    ``major / minor`` axis length - 1 for a circle, larger the more elongated.
    Often easier to read than eccentricity ("3x as long as wide"). Returns
    ``inf`` for a degenerate line (zero minor axis).

    ``space`` selects the coordinate system - see :func:`_ellipse_moments`.
    The module-level :data:`elongation` is this with ``space="geographic"``.
    """
    def _elongation(f):
        l1, l2, _ = _ellipse_moments(f, space)
        if l2 <= 0.0:
            return math.inf if l1 > 0.0 else 1.0
        return float(math.sqrt(l1 / l2))

    _elongation.__name__ = f"elongation_{space}"
    _elongation.__doc__ = (
        f"Equivalent-ellipse aspect ratio major/minor, >= 1 ({space} coordinates)."
    )
    return _elongation


def major_axis_km(f):
    """Major-axis length (km) of the feature's equivalent ellipse.

    ``4 * sqrt(largest second moment)`` on the geographic tangent plane - the
    full length of the fitted ellipse, so it scales like the feature's longest
    extent. Pairs with :func:`minor_axis_km` and :func:`orientation_deg`.
    """
    l1, _, _ = _ellipse_moments(f, "geographic")
    return float(4.0 * math.sqrt(l1 if l1 > 0.0 else 0.0))


def minor_axis_km(f):
    """Minor-axis length (km) of the feature's equivalent ellipse.

    ``4 * sqrt(smallest second moment)`` on the geographic tangent plane.
    """
    _, l2, _ = _ellipse_moments(f, "geographic")
    return float(4.0 * math.sqrt(l2 if l2 > 0.0 else 0.0))


def orientation_deg(f):
    """Orientation of the major axis, degrees counterclockwise from east.

    In (-90, 90]: 0 is east-west (zonal) elongation, +/-90 is north-south
    (meridional). Computed on the geographic tangent plane. Meaningless for a
    near-circular feature (see :data:`eccentricity`), so interpret it together
    with an elongation measure.
    """
    _, _, theta = _ellipse_moments(f, "geographic")
    deg = math.degrees(theta)
    if deg <= -90.0:
        deg += 180.0
    elif deg > 90.0:
        deg -= 180.0
    return float(deg)


#: Equivalent-ellipse eccentricity in [0, 1], geographic (km) coordinates.
#: 0 = round, ->1 = thin line. The default elongation measure.
eccentricity = ellipse_eccentricity(space="geographic")

#: Equivalent-ellipse aspect ratio (major/minor), >= 1, geographic coordinates.
elongation = ellipse_elongation(space="geographic")


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


def legacy_is_complete(f):
    """True if the feature's *bounding box* contains no cross-track edge pixel.

    Reproduces the ``is_complete`` flag of the earlier GPM feature codebase,
    which built a swath-edge mask over the whole scene::

        in_swath = ~np.isnan(near_surf_rain)
        edge = in_swath & binary_dilation(~in_swath)     # scipy default = 4-conn

    and then asked whether any edge pixel fell inside the feature's
    ``scipy.ndimage.find_objects`` bounding box.

    Note what that means: the test is on the **bounding box**, not on the
    feature's own pixels. Features are rarely rectangular, so a feature can be
    marked incomplete because of an edge pixel several cells away from any of
    its members - one real case is a 142-pixel L-shaped feature whose bbox
    corner overlaps the swath edge 6.3 pixels from its nearest pixel.

    Prefer :data:`touches_cross_track_edge` for new work: it asks whether the
    feature itself touches the swath edge, which is almost always the intended
    question. This function exists to reproduce and validate against the old
    results (it agrees on 193/193 features of GPM 2Ku over Africa, 2016-06-01).

    Returns True when the feature is *complete* (does not reach the edge), so
    it is the logical negation of a "touches" statistic.
    """
    r0, r1, c0, c1 = f.bbox
    ny, nx = f.grid_shape
    # One-cell halo so that dilation sees the neighbours of the bbox's own
    # cells. Clipping at the array border reproduces binary_dilation's
    # border_value=0, i.e. off-grid is not treated as unobserved.
    R = slice(np.maximum(r0 - 1, 0), np.minimum(r1 + 1, ny))
    C = slice(np.maximum(c0 - 1, 0), np.minimum(c1 + 1, nx))
    sub = f._ctx.field[R, C]
    not_in_swath = np.isnan(sub)
    structure = ndimage.generate_binary_structure(2, 1)
    edge = (~not_in_swath) & ndimage.binary_dilation(not_in_swath, structure=structure)
    # Crop the halo away, leaving exactly the find_objects bounding box.
    edge = edge[r0 - R.start:r0 - R.start + (r1 - r0),
                c0 - C.start:c0 - C.start + (c1 - c0)]
    return not bool(edge.any())


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


def swath_edge_pixels_in_dominant(f):
    """Swath-edge pixels of the feature's *dominant-strip* portion.

    Counts pixels that (a) lie in the feature's dominant strip - the one holding
    the plurality of its pixels, per :func:`gridfeatures.swath.dominant_swath` -
    and (b) have a 4-neighbour in a *different* strip. Off-grid neighbours do
    not count, matching the ``count_grid_edge=False`` convention used for real
    swaths (:data:`cross_track_edge_pixels`): the domain edge is not a swath
    edge. Requires ``use_swath=True`` in the Config (raises otherwise).

    This is the artificial-swath twin of :data:`cross_track_edge_pixels`, and it
    pairs with the runner's ``px_in_swath`` column to form the same fraction on
    gridded model output as on real retrievals::

        DYAMOND: swath_edge_pixels_in_dominant / px_in_swath
        GPM:     cross_track_edge_pixels       / size

    On GPM everything outside the swath is NaN, so the whole feature *is* its
    in-swath portion and ``size`` is the denominator; here every pixel is in
    some strip, so the dominant strip stands in for "what one overpass saw".
    The result is always <= ``px_in_swath``: both restrict to the same strip.

    Do not substitute :func:`swath_edge_pixels` for this. That one tests each
    pixel against its *own* strip, so for a feature straddling a seam it counts
    pixels on *both* sides - a whole-feature quantity whose value can exceed
    ``px_in_swath`` and push the fraction above 1.

    See :func:`gridfeatures.swath.dominant_swath` for the approximation involved
    in treating the dominant-strip portion as the observed feature.
    """
    swath = f._ctx.swath
    if swath is None:
        raise ValueError(
            "swath_edge_pixels_in_dominant requires use_swath=True in the Config."
        )
    ny, nx = swath.shape
    own, _ = dominant_swath(swath[f.rows, f.cols])
    in_dominant = swath[f.rows, f.cols] == own
    rows = f.rows[in_dominant]
    cols = f.cols[in_dominant]
    is_edge = np.zeros(rows.shape, dtype=bool)
    for dr, dc in _NEIGHBORS:
        rr = rows + dr
        cc = cols + dc
        valid = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        diff = np.zeros(rows.shape, dtype=bool)
        diff[valid] = swath[rr[valid], cc[valid]] != own
        is_edge |= diff
    return int(is_edge.sum())


def swath_edge_fraction_in_dominant(f):
    """Fraction of the dominant-strip portion that sits on a swath seam.

    ``swath_edge_pixels_in_dominant(f) / <pixels in the dominant strip>``, in
    [0, 1]. A convenience for interactive use; for batch output prefer emitting
    the raw counts (the stat above plus the runner's ``px_in_swath`` column) and
    forming the ratio in post-processing, so the threshold stays a choice you
    make later rather than one baked into the CSV.
    """
    idx = f.swath_index
    if idx is None:
        raise ValueError(
            "swath_edge_fraction_in_dominant requires use_swath=True in the Config."
        )
    _, px_in_swath = dominant_swath(idx)
    return swath_edge_pixels_in_dominant(f) / px_in_swath


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
