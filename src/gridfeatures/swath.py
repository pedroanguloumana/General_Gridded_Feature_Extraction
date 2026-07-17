"""Artificial satellite-swath emulation.

Tiles a domain into parallel strips ("swaths") of a given width and equatorial
inclination angle. Each grid cell is assigned an integer swath index. Features
can then record which swath they fall in and whether they cross a swath seam
(and would be clipped by a real swath edge).

The point is *not* to search for a swath that fits each feature; it is simply
to lay a fixed family of strips over the domain and see what falls where.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def swath_index(lats2d, lons2d, width_km, angle_deg, origin=None):
    """Assign each cell an integer swath index.

    The domain is projected to local east/north kilometres with an
    equirectangular projection about a reference point, rotated by
    ``angle_deg``, and binned into strips of ``width_km`` along the direction
    perpendicular to the swath's long axis.

    Parameters
    ----------
    lats2d, lons2d : 2D array-like
        Cell-center latitudes/longitudes (degrees), shape (nlat, nlon).
    width_km : float
        Swath width in kilometres.
    angle_deg : float
        Inclination of the swath long axis relative to the equator (degrees).
        0 gives east-west strips; increasing values tilt the strips.
    origin : (lat, lon) or None
        Projection reference point. Defaults to the domain-mean lat/lon.

    Returns
    -------
    numpy.ndarray
        Integer swath index per cell, shape (nlat, nlon), starting at 0.
    """
    if width_km <= 0:
        raise ValueError("width_km must be positive.")

    lats2d = np.asarray(lats2d, dtype=float)
    lons2d = np.asarray(lons2d, dtype=float)

    if origin is None:
        lat0 = float(np.nanmean(lats2d))
        lon0 = float(np.nanmean(lons2d))
    else:
        lat0, lon0 = origin

    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)

    # Equirectangular projection to local kilometres about the reference point.
    x = EARTH_RADIUS_KM * (np.radians(lons2d) - lon0r) * np.cos(lat0r)  # east
    y = EARTH_RADIUS_KM * (np.radians(lats2d) - lat0r)                  # north

    theta = np.radians(angle_deg)
    # Coordinate perpendicular to strips whose long axis is inclined by theta.
    perp = -x * np.sin(theta) + y * np.cos(theta)

    perp0 = np.nanmin(perp)
    idx = np.floor((perp - perp0) / width_km).astype(int)
    return idx


def dominant_swath(idx):
    """The strip holding the plurality of ``idx``, as ``(swath_id, count)``.

    ``idx`` is a 1D array of per-cell swath indices (e.g. a Feature's
    ``swath_index``). Ties are broken by lowest strip index, since
    :func:`numpy.unique` returns sorted values and ``argmax`` takes the first
    maximum.

    This is the single definition of "dominant strip" in the package. The
    runner's ``swath_id``/``px_in_swath`` columns and
    :func:`gridfeatures.stats.swath_edge_pixels_in_dominant` all go through it,
    which is what guarantees the edge count is a subset of ``px_in_swath``: they
    are talking about the same strip. Do not reimplement the rule elsewhere
    (``scipy.stats.mode``, for instance, breaks ties differently), or the
    edge/px_in_swath fraction can silently exceed 1.

    Approximation, by design: detection runs on the full global field, so a
    feature straddling a seam is detected as one feature and we approximate the
    "observed" feature as its dominant-strip portion. A real single-swath
    overpass would only ever see that portion, but it would also segment within
    the swath, possibly splitting or shrinking the feature differently. Fully
    resolving that would mean re-segmenting within each strip, which this
    package does not do.
    """
    values, counts = np.unique(idx, return_counts=True)
    k = int(np.argmax(counts))
    return int(values[k]), int(counts[k])
