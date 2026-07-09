"""Feature detection: threshold a field and label contiguous regions.

Detection is deliberately a plain function so it can be swapped for a more
sophisticated segmenter (e.g. watershed) later without touching the rest of
the pipeline.
"""

import numpy as np
from scipy import ndimage

# Comparison operators usable for both the primary threshold and core thresholds.
COMPARATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
}


def _get_comparator(comparison):
    try:
        return COMPARATORS[comparison]
    except KeyError:
        raise ValueError(
            f"Unknown comparison {comparison!r}; expected one of {sorted(COMPARATORS)}."
        )


def threshold_mask(field, threshold, comparison=">"):
    """Boolean mask of cells satisfying ``field <comparison> threshold``.

    NaN cells (missing data) are always excluded from the mask.
    """
    field = np.asarray(field, dtype=float)
    op = _get_comparator(comparison)
    mask = op(field, threshold)
    mask &= ~np.isnan(field)
    return mask


def label_features(field, threshold, comparison=">", connectivity=1):
    """Label contiguous above-threshold regions of a 2D field.

    Parameters
    ----------
    field : 2D array-like
        The gridded field (e.g. precipitation rate).
    threshold : float
        Detection threshold.
    comparison : str
        One of ``>``, ``>=``, ``<``, ``<=``.
    connectivity : int
        1 for 4-connectivity (edges), 2 for 8-connectivity (edges + corners).

    Returns
    -------
    (labeled, n) : (numpy.ndarray, int)
        Integer label array (0 = background) and the number of features found.
    """
    mask = threshold_mask(field, threshold, comparison)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n = ndimage.label(mask, structure=structure)
    return labeled, int(n)
