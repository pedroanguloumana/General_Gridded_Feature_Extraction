import numpy as np
import pandas as pd

import gridfeatures as gf
from gridfeatures import stats


def _config(files, output_path=None, **kw):
    return gf.Config(
        files=files,
        variable="precipitation",
        threshold=1.0,
        min_size=2,
        statistics={
            "size_px": stats.size,
            "max_precip": stats.max,
            "total_precip": stats.total,
            "core_10mm_px": stats.core_size(10.0),
        },
        output_path=output_path,
        **kw,
    )


def test_process_file_rows(netcdf_file):
    rows = gf.process_file(netcdf_file, _config(netcdf_file))
    assert len(rows) == 2  # two blobs above min_size=2
    for row in rows:
        assert "source_file" in row
        assert "feature_id" in row
        assert row["size_px"] >= 2
        assert row["source_file"] == netcdf_file


def test_run_writes_csv(netcdf_file, tmp_path):
    out = tmp_path / "out" / "features.csv"
    df = gf.run(_config(netcdf_file, output_path=str(out)))
    assert out.exists()
    disk = pd.read_csv(out)
    assert len(disk) == len(df) == 2
    assert {"size_px", "max_precip", "total_precip", "core_10mm_px"} <= set(df.columns)


def test_provenance_ids_unique(netcdf_file):
    rows = gf.process_file(netcdf_file, _config(netcdf_file))
    ids = [r["feature_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_time_dimension(netcdf_file_with_time):
    config = _config(netcdf_file_with_time, time_name="time")
    rows = gf.process_file(netcdf_file_with_time, config)
    # Two timesteps, each with two qualifying blobs.
    assert len(rows) == 4
    assert {r["time_index"] for r in rows} == {0, 1}
    # Second timestep is scaled x2 -> higher max precip.
    t0max = max(r["max_precip"] for r in rows if r["time_index"] == 0)
    t1max = max(r["max_precip"] for r in rows if r["time_index"] == 1)
    assert t1max > t0max


def test_swath_columns_present(netcdf_file):
    config = _config(netcdf_file, use_swath=True, swath_width_km=150.0, swath_angle_deg=65.0)
    config.statistics["swath_edge_px"] = stats.swath_edge_pixels
    rows = gf.process_file(netcdf_file, config)
    for row in rows:
        assert "swath_id" in row
        assert "crosses_swath_boundary" in row
        assert "n_swaths" in row
        assert "swath_edge_px" in row
        assert row["swath_edge_px"] >= 0


def test_no_swath_columns_when_disabled(netcdf_file):
    rows = gf.process_file(netcdf_file, _config(netcdf_file))
    assert "swath_id" not in rows[0]
