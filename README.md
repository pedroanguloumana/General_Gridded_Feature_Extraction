# gridfeatures

Reusable Python for identifying and characterizing **features** (contiguous
clusters) in gridded climate/meteorological data — precipitation, OLR,
brightness temperature, or any 2D field. Each detected feature becomes one row
of a CSV; each column is a statistic **you** define.

Built to be `pip install`-able from GitHub and dropped into any HPC workflow:
the library holds the reusable detection/statistics logic and stays out of the
way of whatever parallelism you wrap around it.

## Install

```bash
pip install git+https://github.com/pedroanguloumana/General_Gridded_Feature_Extraction.git
```

For local development (editable):

```bash
conda activate general_gridded_feature_extracton
pip install -e ".[test]"
```

## Quickstart

```python
import gridfeatures as gf
from gridfeatures import stats

config = gf.Config(
    files=["dyamond_precip.nc", "gpm_precip.nc"],
    variable="precipitation",
    threshold=1.0,            # mm/hr — contiguous cells above this = one feature
    min_size=4,               # minimum feature size in pixels
    statistics={              # {column_name: function(Feature) -> number}
        "size_px":       stats.size,
        "max_precip":    stats.max,
        "total_precip":  stats.total,
        "core_10mm_px":  stats.core_size(10.0),      # largest >=10 mm/hr core
        "swath_edge_px": stats.swath_edge_pixels,
    },
    use_swath=True,
    swath_width_km=250.0,
    swath_angle_deg=65.0,     # GPM-like inclination
    output_path="features.csv",
)

df = gf.run(config)          # writes features.csv and returns a DataFrame
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a self-contained
runnable version (it synthesizes its own netCDF).

## How detection works

A feature is a contiguous region of cells satisfying
`field <comparison> threshold` (default `>`), found with
`scipy.ndimage.label`. NaNs are treated as missing (never part of a feature).
Set `connectivity=1` for 4-connectivity (default) or `2` for 8-connectivity.
Features smaller than `min_size` **pixels** are dropped.

Detection lives in `gridfeatures.detection.label_features` and is a plain
function, so you can swap in a more sophisticated segmenter later without
touching the rest of the pipeline.

## Writing your own statistics

Each statistic is a function that receives a `Feature` and returns a number.
A `Feature` exposes the member cells and their context:

| attribute | meaning |
|---|---|
| `f.values` | 1D field values at the member cells |
| `f.area`   | 1D per-cell area (km²) |
| `f.lats`, `f.lons` | 1D coordinates of member cells |
| `f.rows`, `f.cols` | integer grid indices |
| `f.size`   | number of pixels |
| `f.mask` / `f.local_mask()` | 2D boolean masks (full grid / bounding box) |
| `f.centroid` | area-weighted (lat, lon) |
| `f.swath_index` | per-cell swath index (or `None`) |
| `f.id`, `f.source`, `f.time` | provenance |

```python
def total_precip(f):
    return (f.values * f.area).sum()

def frac_above_5(f):
    return (f.values > 5).mean()
```

### Built-in statistics (`gridfeatures.stats`)

- `size` — feature size in pixels
- `max` / `maximum`, `min` / `minimum`, `mean`
- `total` — area-weighted sum (e.g. total precip volume)
- `area_km2`
- `centroid_lat`, `centroid_lon`
- `touches_boundary` — `True` if feature touches the grid edge or a NaN cell
- `boundary_pixels` — count of feature pixels bordering the grid edge or a NaN cell
- `touches_cross_track_edge`, `cross_track_edge_pixels` — same, but **ignoring the grid
  edge** (see below); the right choice for real satellite swath data
- `boundary_pixels_where(count_grid_edge=True)`, `touches_boundary_where(...)` —
  **factories** behind the four names above
- `swath_edge_pixels` — number of feature pixels bordering an *artificial*-swath seam

### Grid edges are not always swath edges

NaN 4-neighbours always mark the observable boundary. Whether an **off-grid** 4-neighbour
does is a property of your data, so it is a switch:

```python
stats.boundary_pixels           # == boundary_pixels_where(count_grid_edge=True)
stats.cross_track_edge_pixels   # == boundary_pixels_where(count_grid_edge=False)
```

Use `count_grid_edge=True` when the array edge really is a data boundary — a model domain
edge, a cutout you mean to treat as closed.

Use `count_grid_edge=False` for satellite swath crops such as **GPM L2**. There the array
edges are the *along-track* cut (the granule's time range) or a regional box clip, and the
instrument observed straight through them; only NaN cells inside the grid mark the real
*cross-track* swath edge. Counting grid edges inflates the result by the entire
along-track cap — on one day of GPM 2Ku over Africa it over-flagged 40 of 252 features
where the correct answer was 34.

`count_grid_edge=False` is slightly conservative: where a cross-track edge exits through a
corner of a tight crop, the unobserved neighbour is off-grid rather than NaN, so a few
pixels per swath are missed. Fixing that would need native along/cross scan coordinates.
- `core_size(core_threshold, comparison=">=", connectivity=None)` — **factory**;
  size (pixels) of the largest contiguous sub-region above a higher threshold
  (e.g. the biggest 10 mm/hr core inside a 1 mm/hr feature)

## Artificial satellite swaths

To emulate a GPM-style swath over a global model field, set `use_swath=True`.
The domain is tiled into parallel strips of `swath_width_km` inclined
`swath_angle_deg` from the equator; each cell gets a swath index. Every feature
then gets extra CSV columns:

- `swath_id` — the dominant swath the feature falls in
- `n_swaths` — how many swaths it spans
- `crosses_swath_boundary` — `True` if it spans more than one (would be clipped)

This intentionally lays a fixed set of strips over the domain and records what
falls where; it does not search for a best-fitting swath per feature.

## Output columns

Every row carries provenance plus your statistics:

- `source_file` — originating file path
- `feature_id` — `<basename>:<time_index>:<label>`, unique per feature
- `time`, `time_index` — present when `time_name` is set
- swath columns — present when `use_swath=True`
- one column per entry in `statistics`

## Parallelism

`gridfeatures` deliberately contains **no** parallel machinery. The natural
unit of work is `gridfeatures.process_file(path, config) -> list[dict]`; map it
over files with your scheduler / `multiprocessing` / dask and concatenate the
rows.

## Scope / assumptions (v0.1)

- Regular (rectilinear) lat/lon grids read via xarray; per-cell area from lat
  spacing. Native curvilinear GPM L2 swath coordinates are not yet handled.
- Centroid longitude is a plain weighted mean (no antimeridian handling).

## Development

```bash
conda activate general_gridded_feature_extracton
pip install -e ".[test]"
pytest
```
