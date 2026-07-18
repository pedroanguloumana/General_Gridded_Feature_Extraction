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
conda activate general_gridded_feature_extraction
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
- `north_extent`, `south_extent`, `east_extent`, `west_extent` — the feature's
  bounding coordinates (deg); `lat_extent`, `lon_extent` — the spans. Longitude is
  **antimeridian-aware** (see below)
- `eccentricity`, `elongation` — shape of the equivalent fitted ellipse, for
  describing how elongated a feature is (see below)
- `major_axis_km`, `minor_axis_km`, `orientation_deg` — the fitted ellipse's axis
  lengths (km) and major-axis orientation (deg CCW from east)
- `ellipse_eccentricity(space=...)`, `ellipse_elongation(space=...)` — **factories**
  behind `eccentricity`/`elongation`; `space="geographic"` (default, km tangent plane)
  or `space="grid"` (raw pixel indices, matching `regionprops`)
- `touches_boundary` — `True` if feature touches the grid edge or a NaN cell
- `boundary_pixels` — count of feature pixels bordering the grid edge or a NaN cell
- `touches_cross_track_edge`, `cross_track_edge_pixels` — same, but **ignoring the grid
  edge** (see below); the right choice for real satellite swath data
- `boundary_pixels_where(count_grid_edge=True)`, `touches_boundary_where(...)` —
  **factories** behind the four names above
- `swath_edge_pixels` — number of feature pixels bordering an *artificial*-swath seam,
  counted over the **whole** feature (each pixel against its own strip)
- `swath_edge_pixels_in_dominant` — same, but restricted to the feature's **dominant
  strip**; pairs with the `px_in_swath` column to measure how much of an artificial-swath
  feature sits against a seam (see below)
- `swath_edge_fraction_in_dominant` — the ratio of the two, as a convenience
- `legacy_is_complete` — bounding-box edge test, for reproducing prior GPM results
  (see below)

### Extent and elongation (ellipse fitting)

Four cardinal extents record where each feature reaches, and an equivalent-ellipse fit
describes its shape:

```python
config.statistics.update({
    "north": stats.north_extent,   "south": stats.south_extent,
    "east":  stats.east_extent,    "west":  stats.west_extent,
    "lon_span_deg":  stats.lon_extent,
    "eccentricity":  stats.eccentricity,   # 0 = round, ->1 = thin line
    "elongation":    stats.elongation,     # major/minor axis ratio, >= 1
    "major_km":      stats.major_axis_km,
    "orientation":   stats.orientation_deg,  # 0 = zonal (E-W), +/-90 = meridional
})
```

The ellipse is fit from the feature footprint's second moments (every member cell counts
once, unweighted). By default it is computed on a **local east/north tangent plane in
km** (`space="geographic"`), so elongation reflects true ground distance — grid cells are
not square in km away from the equator, and a purely index-space fit would over- or
under-state elongation by up to `1/cos(lat)`. Pass `stats.ellipse_eccentricity(space="grid")`
for the raw-pixel `regionprops` convention instead.

`eccentricity` and `elongation` are two views of the same ratio: eccentricity is bounded
in `[0, 1)` (handy for thresholding), while elongation reads directly as "N times as long
as wide". `orientation_deg` is only meaningful for a feature that is actually elongated —
interpret it alongside one of the shape ratios.

**Antimeridian.** The longitude extents and the ellipse fit heal the ±180° seam: the
member longitudes are unrolled across the largest empty gap in the feature's longitude
arc, so a feature straddling the dateline reports its true (narrow) `lon_extent` rather
than ~360°. A straddling feature's `east_extent` comes out numerically *less* than its
`west_extent` — that inversion is the signal that the feature wraps past +180°. The
heuristic assumes a feature spans well under 180° of longitude, which holds for any real
contiguous feature. (`centroid_lon` is unchanged and still naive at the seam.)

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

### `legacy_is_complete`: bounding box vs. pixels

An earlier GPM feature codebase built a scene-wide edge mask and then asked whether any
edge pixel fell inside a feature's `scipy.ndimage.find_objects` **bounding box**:

```python
in_swath = ~np.isnan(near_surf_rain)
edge = in_swath & binary_dilation(~in_swath)     # == touches_cross_track_edge semantics
is_complete = not edge[find_objects(labeled)[i]].any()
```

The edge mask itself is exactly what `cross_track_edge_pixels` counts. The difference is
that it is intersected with the bounding box rather than with the feature's own pixels.
Features are rarely rectangular, so this marks a feature incomplete when an edge pixel
merely falls within its bbox — in one real case a 142-pixel L-shaped feature whose bbox
corner overlaps the swath edge **6.3 pixels** from its nearest member pixel.

`stats.legacy_is_complete` reproduces that flag exactly (193/193 features, GPM 2Ku over
Africa, 2016-06-01) and returns `True` when the feature is complete. Use it to validate
against old results; prefer `touches_cross_track_edge` for new work, which asks whether
the feature *itself* reaches the swath edge.
- `core_size(core_threshold, comparison=">=", connectivity=None)` — **factory**;
  size (pixels) of the largest contiguous sub-region above a higher threshold
  (e.g. the biggest 10 mm/hr core inside a 1 mm/hr feature)

## Artificial satellite swaths

To emulate a GPM-style swath over a global model field, set `use_swath=True`.
The domain is tiled into parallel strips of `swath_width_km` inclined
`swath_angle_deg` from the equator; each cell gets a swath index. Every feature
then gets extra CSV columns:

- `swath_id` — the dominant swath the feature falls in (the strip holding the
  plurality of its pixels; ties go to the lowest strip index)
- `px_in_swath` — how many of the feature's pixels are in that dominant strip
- `n_swaths` — how many swaths it spans
- `crosses_swath_boundary` — `True` if it spans more than one (would be clipped)

This intentionally lays a fixed set of strips over the domain and records what
falls where; it does not search for a best-fitting swath per feature.

### Keeping features that are mostly within the swath

The same question — *is this feature mostly inside the swath, or is it clipped?* —
has to be asked of gridded model output and of real retrievals in a way that gives
comparable answers. The fraction is the same on both sides:

```
fraction = (swath-edge pixels of the feature) / (feature pixels within the swath)
```

but each term is spelled differently, because "within the swath" means something
different:

| | model field + artificial swath | real swath (e.g. GPM L2) |
|---|---|---|
| within the swath | `px_in_swath` column — pixels in the dominant strip | `stats.size` — everything outside the swath is NaN, so the whole feature *is* the in-swath portion |
| swath edge | `stats.swath_edge_pixels_in_dominant` — dominant-strip pixels with a 4-neighbour in another strip | `stats.cross_track_edge_pixels` — pixels bordering NaN, grid edges ignored |

Neither numerator counts off-grid neighbours, so the domain edge is never mistaken
for a swath edge on either side.

Both terms are raw counts, so the threshold stays a post-processing choice:

```python
config.statistics["swath_edge_px_dom"] = stats.swath_edge_pixels_in_dominant
df = gridfeatures.run(config)
mostly_inside = df[df.swath_edge_px_dom / df.px_in_swath < 0.05]
```

On the real-swath side, add `stats.size` and `stats.cross_track_edge_pixels` to
`statistics` and take the same ratio. (`stats.swath_edge_fraction_in_dominant`
returns the model-side ratio directly if you want it in one column.)

Two things to know about the artificial-swath side:

- Use `swath_edge_pixels_in_dominant`, not `swath_edge_pixels`, as the numerator.
  The latter tests each pixel against *its own* strip, so a feature straddling a
  seam contributes pixels from both sides — a whole-feature count that can exceed
  `px_in_swath` and push the fraction above 1.
- Detection runs on the full field, so a feature spanning strips is detected as
  one feature and its dominant-strip portion stands in for what a single overpass
  would have seen. That is an approximation: a real overpass would also segment
  *within* the swath. Re-segmenting per strip is out of scope here.

## Output columns

Every row carries provenance plus your statistics:

- `source_file` — originating file path
- `feature_id` — `<basename>:<time_index>:<label>`, unique per feature
- `time`, `time_index` — present when `time_name` is set
- swath columns (`swath_id`, `px_in_swath`, `n_swaths`, `crosses_swath_boundary`) —
  present when `use_swath=True`
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
conda activate general_gridded_feature_extraction
pip install -e ".[test]"
pytest
```
