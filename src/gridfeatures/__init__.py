"""gridfeatures - identify and characterize features in gridded climate data.

Quickstart
----------
>>> import gridfeatures as gf
>>> from gridfeatures import stats
>>> config = gf.Config(
...     files="dyamond_precip.nc",
...     variable="precipitation",
...     threshold=1.0,                     # mm/hr, defines features
...     min_size=4,                        # pixels
...     statistics={
...         "size_px": stats.size,
...         "max_precip": stats.max,
...         "total_precip": stats.total,
...         "core_10mm_px": stats.core_size(10.0),
...         "swath_edge_px": stats.swath_edge_pixels,
...     },
...     use_swath=True,
...     swath_width_km=250.0,
...     swath_angle_deg=65.0,
...     output_path="features.csv",
... )
>>> df = gf.run(config)
"""

from . import detection, grid, stats, swath
from .config import Config
from .detection import label_features
from .feature import Feature, FieldContext
from .runner import extract_features, feature_row, process_file, run

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Feature",
    "FieldContext",
    "run",
    "process_file",
    "extract_features",
    "feature_row",
    "label_features",
    "stats",
    "detection",
    "swath",
    "grid",
    "__version__",
]
