import numpy as np

from gridfeatures import detection


def test_label_counts_blobs(simple_field):
    field, _, _ = simple_field
    labeled, n = detection.label_features(field, threshold=1.0, comparison=">")
    # Blob A, Blob B, and the tiny 1-cell blob = 3 features.
    assert n == 3
    assert labeled.max() == 3


def test_nan_excluded():
    field = np.array([[5.0, np.nan], [5.0, 5.0]])
    mask = detection.threshold_mask(field, 1.0, ">")
    assert mask[0, 1] == False  # noqa: E712 - NaN never in a feature
    assert mask.sum() == 3


def test_connectivity_diagonal():
    # Two cells touching only at a corner.
    field = np.array([[5.0, 0.0], [0.0, 5.0]])
    _, n4 = detection.label_features(field, 1.0, ">", connectivity=1)
    _, n8 = detection.label_features(field, 1.0, ">", connectivity=2)
    assert n4 == 2
    assert n8 == 1


def test_comparison_less_than():
    field = np.array([[0.0, 0.0], [5.0, 5.0]])
    _, n = detection.label_features(field, 1.0, comparison="<")
    assert n == 1
