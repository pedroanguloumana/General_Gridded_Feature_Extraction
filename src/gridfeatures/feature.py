"""The :class:`Feature` object handed to every user statistic function."""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FieldContext:
    """Everything a Feature needs to describe itself and its surroundings.

    A single context is shared by all features detected in one 2D field, so
    per-feature objects stay lightweight.
    """

    field: np.ndarray            # 2D field values
    lats2d: np.ndarray           # 2D latitudes (deg north)
    lons2d: np.ndarray           # 2D longitudes (deg east)
    area: np.ndarray             # 2D per-cell area (km^2)
    labeled: np.ndarray          # 2D integer labels (0 = background)
    swath: Optional[np.ndarray]  # 2D integer swath index, or None if disabled
    source: str                  # provenance: originating file path
    time: object                 # time value of this slice (or None)
    time_index: Optional[int]    # time index of this slice (or None)
    connectivity: int            # connectivity used for detection


class Feature:
    """A single detected feature (contiguous cluster of cells).

    User statistic functions receive one of these and typically reduce over the
    member cells, e.g.::

        def total_precip(f):
            return (f.values * f.area).sum()
    """

    def __init__(self, label, ctx):
        self.label = int(label)
        self._ctx = ctx
        rows, cols = np.where(ctx.labeled == self.label)
        self.rows = rows
        self.cols = cols
        self._r0, self._r1 = int(rows.min()), int(rows.max()) + 1
        self._c0, self._c1 = int(cols.min()), int(cols.max()) + 1

    # -- provenance -------------------------------------------------------
    @property
    def source(self):
        """Originating file path."""
        return self._ctx.source

    @property
    def time(self):
        """Time value of the slice this feature came from (or None)."""
        return self._ctx.time

    @property
    def time_index(self):
        """Time index of the slice this feature came from (or None)."""
        return self._ctx.time_index

    @property
    def id(self):
        """Stable identifier: ``<basename>:<time_index>:<label>``."""
        base = os.path.basename(self._ctx.source) if self._ctx.source else "field"
        parts = [base]
        if self._ctx.time_index is not None:
            parts.append(str(self._ctx.time_index))
        parts.append(str(self.label))
        return ":".join(parts)

    # -- geometry / member-cell data -------------------------------------
    @property
    def grid_shape(self):
        return self._ctx.field.shape

    @property
    def size(self):
        """Number of cells/pixels in the feature."""
        return int(self.rows.size)

    @property
    def values(self):
        """1D field values at the member cells."""
        return self._ctx.field[self.rows, self.cols]

    @property
    def lats(self):
        return self._ctx.lats2d[self.rows, self.cols]

    @property
    def lons(self):
        return self._ctx.lons2d[self.rows, self.cols]

    @property
    def area(self):
        """1D per-cell area (km^2) at the member cells."""
        return self._ctx.area[self.rows, self.cols]

    @property
    def swath_index(self):
        """1D artificial-swath index at the member cells, or None if disabled."""
        if self._ctx.swath is None:
            return None
        return self._ctx.swath[self.rows, self.cols]

    @property
    def bbox(self):
        """Bounding box as (row_start, row_stop, col_start, col_stop)."""
        return (self._r0, self._r1, self._c0, self._c1)

    @property
    def mask(self):
        """Full-grid 2D boolean mask of the feature (constructed on demand)."""
        m = np.zeros(self._ctx.field.shape, dtype=bool)
        m[self.rows, self.cols] = True
        return m

    def local_mask(self):
        """Boolean mask of the feature within its bounding box (cheap)."""
        sub = self._ctx.labeled[self._r0:self._r1, self._c0:self._c1]
        return sub == self.label

    def local_field(self):
        """Field values within the feature's bounding box."""
        return self._ctx.field[self._r0:self._r1, self._c0:self._c1]

    @property
    def centroid(self):
        """Area-weighted (lat, lon) centroid.

        Note: longitude is a simple weighted mean and does not handle the
        antimeridian; keep domains away from the +/-180 seam or work in a
        shifted longitude convention.
        """
        w = self.area
        wsum = w.sum()
        return (
            float((self.lats * w).sum() / wsum),
            float((self.lons * w).sum() / wsum),
        )

    def __repr__(self):
        return f"Feature(id={self.id!r}, size={self.size})"
