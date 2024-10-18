from enum import Enum
from math import inf
from typing import Tuple

import numpy as np
from xarray import Dataset

from openlifu.bf.offset_grid import offset_grid
from openlifu.geo import Point
from openlifu.util.units import get_ndgrid_from_arr, rescale_coords

MaskOp = Enum("MaskOp", ["GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL"])


def mask_focus(
        data_arr: Dataset,
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
        data_arr : xarray.Dataset
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
    # The distance is calculated by first transforming the coordinate system so that the focus point is on
    # the z' axis, adjusting the x and y axes to be orthogonal to the z' axis, and then calculating the distance
    # e.g. d = sqrt(((x'-x0')/ax)^2 + ((y'-y0')/ay)^2 + ((z'-z0')/az)^2). This is useful for calculating how far
    # away from an oblong focal spot each point is.
    ogrid = offset_grid(data_arr_rescaled, focus)
    ogrid_aspect_corrected = ogrid*aspect_ratio
    m = np.sqrt(np.sum(ogrid_aspect_corrected**2, axis=-1))

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
