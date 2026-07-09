"""Shared test fixtures."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def simple_field():
    """A 6x8 field with two well-separated blobs and a tiny one-cell blob.

    Returns (field, lats, lons).
    """
    field = np.zeros((6, 8), dtype=float)
    # Blob A: 2x2 block, peak 12 at one corner.
    field[1:3, 1:3] = 5.0
    field[1, 1] = 12.0
    # Blob B: 2x3 block, all 3.0 (never reaches a 10 core).
    field[4:6, 5:8] = 3.0
    # Tiny 1-cell blob (below a min_size of 2).
    field[0, 6] = 8.0

    lats = np.linspace(-2.5, 2.5, 6)   # deg north
    lons = np.linspace(0.0, 7.0, 8)    # deg east
    return field, lats, lons


@pytest.fixture
def netcdf_file(tmp_path, simple_field):
    """Write simple_field to a netCDF file and return its path."""
    field, lats, lons = simple_field
    ds = xr.Dataset(
        {"precipitation": (("lat", "lon"), field)},
        coords={"lat": lats, "lon": lons},
    )
    path = tmp_path / "precip.nc"
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def netcdf_file_with_time(tmp_path, simple_field):
    """Write a 2-timestep netCDF file (second step scaled up)."""
    field, lats, lons = simple_field
    stack = np.stack([field, field * 2.0], axis=0)
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), stack)},
        coords={"time": [0, 1], "lat": lats, "lon": lons},
    )
    path = tmp_path / "precip_time.nc"
    ds.to_netcdf(path)
    return str(path)
