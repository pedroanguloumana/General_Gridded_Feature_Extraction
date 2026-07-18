# Demonstration notebooks

Short, self-contained notebooks that show the tricky parts of `gridfeatures`
behaving correctly on **synthetic data with a known answer**. Every notebook
builds its own field, so there is nothing to download, and each ends in `assert`
checks — the plots are the intuition, the assertions are the proof. They are
committed **with their outputs**, so you can read them on GitHub without running
anything.

| Notebook | Shows |
|---|---|
| [`01_feature_extraction_basics.ipynb`](01_feature_extraction_basics.ipynb) | Detection, cardinal extents, and the ellipse-fit elongation stats — a disk, a tilted ellipse, and a near-line, with the *fitted* ellipse drawn back over each footprint. |
| [`02_antimeridian.ipynb`](02_antimeridian.ipynb) | A feature straddling the ±180° dateline. A naive longitude span reads ~360°; the package heals the seam and recovers the true narrow extent and shape, identical to the same feature far from the seam. |
| [`03_artificial_swaths.ipynb`](03_artificial_swaths.ipynb) | Tiling the domain into artificial satellite swaths, the dominant-strip rule, and the seam-edge counts — including why the whole-feature edge count differs from the dominant-strip one that pairs with `px_in_swath`. |
| [`04_swath_edge_semantics.ipynb`](04_swath_edge_semantics.ipynb) | Real-retrieval edge semantics: NaN cross-track edges vs. the array (grid) edge, why an along-track cap is *not* a swath edge, and the legacy bounding-box `is_complete` flag. |

## Running them

The notebooks need the plotting/notebook extras on top of the package:

```bash
pip install -e ".[notebooks]"
```

They are written for the project's Jupyter kernel
(`Python (general_gridded_feature_extraction)`); select that kernel, or just
"Run All" with any kernel that has the package installed. To re-execute headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
