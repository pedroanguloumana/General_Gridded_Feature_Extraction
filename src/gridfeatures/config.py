"""User-facing configuration object."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union


@dataclass
class Config:
    """All settings for a feature-extraction run.

    Parameters
    ----------
    files : str or list of str
        netCDF file(s) to process.
    variable : str
        Name of the variable to extract from each file (e.g. ``"precipitation"``).
    threshold : float
        Detection threshold; contiguous cells satisfying ``comparison`` become
        one feature.
    statistics : dict
        Mapping of ``{column_name: function}`` where each function takes a
        :class:`~gridfeatures.feature.Feature` and returns a number. Determines
        the CSV columns.
    min_size : int
        Minimum feature size in pixels/cells; smaller features are dropped.
    comparison : str
        Detection comparison operator: ``>``, ``>=``, ``<`` or ``<=``.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity.
    output_path : str or None
        Where to write the CSV. If None, :func:`~gridfeatures.runner.run` skips
        writing and just returns the DataFrame.
    use_swath : bool
        If True, overlay artificial swaths and add swath columns.
    swath_width_km : float
        Swath strip width in kilometres.
    swath_angle_deg : float
        Swath inclination relative to the equator (degrees).
    lat_name, lon_name : str
        Coordinate variable names in the netCDF file.
    time_name : str or None
        Time dimension name; if set, each time step is processed independently.
    """

    files: Union[str, List[str]]
    variable: str
    threshold: float
    statistics: Dict[str, Callable]

    min_size: int = 1
    comparison: str = ">"
    connectivity: int = 1
    output_path: Optional[str] = "features.csv"

    use_swath: bool = False
    swath_width_km: float = 250.0
    swath_angle_deg: float = 0.0

    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.files, str):
            self.files = [self.files]
        if not self.statistics:
            raise ValueError("Config.statistics must contain at least one column.")
        if self.min_size < 1:
            raise ValueError("min_size must be >= 1 (pixels).")
