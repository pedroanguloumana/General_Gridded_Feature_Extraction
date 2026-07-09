"""Minimal end-to-end example.

Generates a small synthetic precipitation netCDF (so it runs with no external
data), then extracts features and writes a CSV - the same pattern you'd use for
real GPM / DYAMOND files by swapping in your file paths and variable name.

Run:  python examples/basic_usage.py
"""

import numpy as np
import xarray as xr

import gridfeatures as gf
from gridfeatures import stats


def make_demo_netcdf(path):
    rng = np.random.default_rng(0)
    lats = np.linspace(-10, 10, 80)
    lons = np.linspace(0, 40, 160)
    field = np.zeros((lats.size, lons.size))
    # Drop a few gaussian "storms" with intense cores.
    for (clat, clon, amp, width) in [(-3, 8, 25, 1.5), (4, 20, 40, 1.3), (0, 32, 15, 1.8)]:
        la = lats[:, None]
        lo = lons[None, :]
        field += amp * np.exp(-(((la - clat) ** 2 + (lo - clon) ** 2) / (2 * width**2)))
    field += rng.normal(0, 0.1, field.shape).clip(0)
    ds = xr.Dataset(
        {"precipitation": (("lat", "lon"), field)},
        coords={"lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)


def main():
    make_demo_netcdf("demo_precip.nc")

    config = gf.Config(
        files="demo_precip.nc",
        variable="precipitation",
        threshold=1.0,           # mm/hr defines a feature
        min_size=4,              # pixels
        statistics={
            "size_px": stats.size,
            "max_precip": stats.max,
            "total_precip": stats.total,
            "core_10mm_px": stats.core_size(10.0),   # largest >=10 mm/hr core
            "swath_edge_px": stats.swath_edge_pixels,
            "touches_domain_edge": stats.touches_boundary,
        },
        use_swath=True,
        swath_width_km=250.0,
        swath_angle_deg=65.0,    # GPM-like inclination
        output_path="features.csv",
    )

    df = gf.run(config)
    print(df.to_string(index=False))
    print("\nWrote features.csv")


if __name__ == "__main__":
    main()
