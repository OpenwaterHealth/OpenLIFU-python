import numpy as np
from typing import Tuple
from math import inf
from enum import Enum

from xarray import DataArray

from openlifu.util.units import rescale_coords, get_ndgrid_from_arr
from openlifu.geo import Point
from openlifu.bf import calc_dist_from_focus

MaskOp = Enum("MaskOp", ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL"])


def mask_focus(
        data_arr: DataArray,
        focus: Point,
        distance: float,
        mask_op: MaskOp = MaskOp.LESS_EQUAL,
        units: str = "m",
        aspect_ratio: Tuple[float, float, float] = (1., 1., 10.),
        z_min: float = -inf
        ) -> np.ndarray:
    """
    Creates a mask for points within a (scaled) distance from the focus point.

    Args:
        data_arr : xarray.DataArray
            The input DataArray.
        focus : fus.Point object
            The focus point to be used as reference for masking.
        distance : float
            The maximum allowed distance in units defined by 'units' option.
        units : str
            Distance units (Default: "m").
        mask_op : str
            Operation to perform on the mask (Default: "LESS_EQUAL").
        aspect_ratio : Tuple[float, float, float]
            Aspect ratio to calculate distance (Default: (1, 1, 10)).
        z_min : float
            Minimum z-value for masking points below it (Default: -inf).

    Returns:
        A boolean array indicating which points are within the specified range.
    """
    # Rescale coordinates and focus to the specified units
    data_arr_rescaled = rescale_coords(data_arr, units)
    focus.rescale(units)

    # Calculate distances from the focus for each point in coords and compare with distance limit
    m = calc_dist_from_focus(data_arr_rescaled, focus, aspect_ratio)

    # mask based on distance
    if mask_op is MaskOp.GREATER:
        mask = np.greater(m, distance)
    elif mask_op is MaskOp.GREATER_EQUAL:
        mask = np.greater_equal(m, distance)
    elif mask_op is MaskOp.LESS:
        mask = np.less(m, distance)
    elif mask_op is MaskOp.LESS_EQUAL:
        mask = np.less_equal(m, distance)
    else:
        raise ValueError(f"Mask operation {mask_op} is not defined!")
    if z_min > -inf:
        XYZ = get_ndgrid_from_arr(data_arr_rescaled)
        zmask = XYZ[..., 2] > z_min
        mask &= zmask

    return mask
