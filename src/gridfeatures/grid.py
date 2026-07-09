"""Grid geometry helpers for regular (rectilinear) lat/lon grids."""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def _edges_from_centers(x):
    """Cell edges inferred from 1D cell-center coordinates."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two coordinate values to infer cell edges.")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def cell_area_km2(lats, lons):
    """Per-cell area in km^2 for a rectilinear lat/lon grid.

    Parameters
    ----------
    lats, lons : 1D array-like
        Cell-center latitudes (degrees north) and longitudes (degrees east).

    Returns
    -------
    numpy.ndarray
        Array of shape (nlat, nlon) with each cell's area in km^2.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lat_edges = np.radians(_edges_from_centers(lats))
    lon_edges = np.radians(_edges_from_centers(lons))
    # Area of a lat/lon cell = R^2 * |sin(lat2) - sin(lat1)| * |lon2 - lon1|.
    dsin = np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))  # (nlat,)
    dlon = np.abs(np.diff(lon_edges))                              # (nlon,)
    return (EARTH_RADIUS_KM ** 2) * np.outer(dsin, dlon)
