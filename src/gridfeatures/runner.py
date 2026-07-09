"""Orchestration: field -> features -> rows -> CSV.

The functions here are intended to be composable. External code that handles
parallelism can map :func:`process_file` over many files; this module keeps the
reusable per-file / per-field logic and stays free of any parallel machinery.
"""

import numpy as np
import pandas as pd

from .detection import label_features
from .feature import Feature, FieldContext
from .grid import cell_area_km2
from .io import read_fields, write_csv
from .swath import swath_index


def _area_from_2d(lats2d, lons2d):
    """Per-cell area (km^2) for a rectilinear grid given 2D coordinate arrays."""
    lats1d = lats2d[:, 0]
    lons1d = lons2d[0, :]
    return cell_area_km2(lats1d, lons1d)


def extract_features(
    field,
    lats2d,
    lons2d,
    config,
    source,
    time=None,
    time_index=None,
    area=None,
    swath=None,
):
    """Detect and filter features in a single 2D field.

    Returns a list of :class:`~gridfeatures.feature.Feature`. Features smaller
    than ``config.min_size`` pixels are dropped.
    """
    field = np.asarray(field, dtype=float)
    if area is None:
        area = _area_from_2d(lats2d, lons2d)

    labeled, n = label_features(
        field, config.threshold, config.comparison, config.connectivity
    )
    ctx = FieldContext(
        field=field,
        lats2d=lats2d,
        lons2d=lons2d,
        area=area,
        labeled=labeled,
        swath=swath,
        source=source,
        time=time,
        time_index=time_index,
        connectivity=config.connectivity,
    )

    features = []
    if n == 0:
        return features

    counts = np.bincount(labeled.ravel())
    for label in range(1, n + 1):
        if counts[label] < config.min_size:
            continue
        features.append(Feature(label, ctx))
    return features


def feature_row(feature, config):
    """Build one output row (dict) for a feature: provenance + swath + stats."""
    row = {
        "source_file": feature.source,
        "feature_id": feature.id,
    }
    if feature.time is not None:
        row["time"] = feature.time
    if feature.time_index is not None:
        row["time_index"] = feature.time_index

    if config.use_swath and feature.swath_index is not None:
        idx = feature.swath_index
        values, counts = np.unique(idx, return_counts=True)
        row["swath_id"] = int(values[np.argmax(counts)])   # dominant swath
        row["n_swaths"] = int(values.size)
        row["crosses_swath_boundary"] = bool(values.size > 1)

    for column, fn in config.statistics.items():
        row[column] = fn(feature)
    return row


def process_file(path, config):
    """Process one netCDF file into a list of feature rows.

    This is the natural unit of work to parallelize externally (one call per
    file, or shard the file list across workers).
    """
    rows = []
    for field, lats2d, lons2d, tval, tidx in read_fields(
        path, config.variable, config.lat_name, config.lon_name, config.time_name
    ):
        area = _area_from_2d(lats2d, lons2d)
        swath = None
        if config.use_swath:
            swath = swath_index(
                lats2d, lons2d, config.swath_width_km, config.swath_angle_deg
            )
        features = extract_features(
            field,
            lats2d,
            lons2d,
            config,
            source=path,
            time=tval,
            time_index=tidx,
            area=area,
            swath=swath,
        )
        rows.extend(feature_row(f, config) for f in features)
    return rows


def run(config, write=True):
    """Process all files in ``config`` and return a :class:`pandas.DataFrame`.

    If ``write`` is True and ``config.output_path`` is set, the CSV is written
    as a side effect.
    """
    all_rows = []
    for path in config.files:
        all_rows.extend(process_file(path, config))

    df = pd.DataFrame(all_rows)
    if write and config.output_path:
        write_csv(all_rows, config.output_path)
    return df
