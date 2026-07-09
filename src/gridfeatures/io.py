"""I/O: read gridded fields from netCDF (via xarray) and write feature CSVs."""

import os

import numpy as np
import pandas as pd
import xarray as xr


def _make_2d(lats, lons):
    """Broadcast 1D lat/lon centers to 2D (nlat, nlon) grids.

    Already-2D coordinates are returned unchanged.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.ndim == 1 and lons.ndim == 1:
        return np.meshgrid(lats, lons, indexing="ij")
    return lats, lons


def _order_field(da, lat_name, lon_name):
    """Return the field as a 2D (lat, lon) float array."""
    da = da.transpose(lat_name, lon_name)
    return np.asarray(da.values, dtype=float)


def read_fields(path, variable, lat_name="lat", lon_name="lon", time_name=None):
    """Yield 2D field slices from a netCDF file.

    Yields
    ------
    (field, lats2d, lons2d, time_value, time_index)
        ``field`` is a 2D (lat, lon) array. If ``time_name`` is given and
        present, one slice is yielded per time step with its value/index;
        otherwise a single slice is yielded with ``(None, None)``.
    """
    ds = xr.open_dataset(path)
    try:
        da = ds[variable]
        lats2d, lons2d = _make_2d(ds[lat_name].values, ds[lon_name].values)

        if time_name is not None and time_name in da.dims:
            times = ds[time_name].values
            for i in range(da.sizes[time_name]):
                slab = da.isel({time_name: i})
                field = _order_field(slab, lat_name, lon_name)
                yield field, lats2d, lons2d, times[i], i
        else:
            field = _order_field(da, lat_name, lon_name)
            yield field, lats2d, lons2d, None, None
    finally:
        ds.close()


def write_csv(rows, path):
    """Write a list of row dicts to CSV, creating parent directories as needed.

    Returns the resulting :class:`pandas.DataFrame`.
    """
    df = pd.DataFrame(rows)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    return df
