from typing import Tuple

import numpy as np
from xarray import DataArray

from openlifu.bf.offset_grid import offset_grid
from openlifu.geo import Point


def calc_dist_from_focus(
        data_arr: DataArray,
        focus: Point,
        # units: str = "m", #TODO: I removed since input is already rescaled coords (already check were it is used)
        aspect_ratio: Tuple[float, float, float] = (1., 1., 1.)
        ) -> np.ndarray:
    """
    Calculates the distance from the focus point for each point in the coordinate system.

    When an aspect ratio is provided, the linear distances are divided by the aspect ratio.
    The distance is calculated by first transforming the coordinate system so that the focus point is on
    the z' axis, adjusting the x and y axes to be orthogonal to the z' axis, and then calculating the distance
    e.g. d = sqrt(((x'-x0')/ax)^2 + ((y'-y0')/ay)^2 + ((z'-z0')/az)^2). This is useful for calculating how far
    away from an oblong focal spot each point is.

    Args:
        data_arr: xarray.DataArray
        focus : fus.Point object
            The focus point to be used as reference.
        aspect_ratio : Tuple[float, float, float]
            Aspect ratio to calculate distance (Default: (1., 1., 1.)).

    Returns:
        Array of distances from focus point.
    """
    # Calculate offset grid
    ogrid = offset_grid(data_arr, focus)

    # Calculate distance from focus point
    ogrid_aspect_corrected = ogrid*aspect_ratio
    dist_from_focus = np.sqrt(np.sum(ogrid_aspect_corrected**2, axis=-1))

    return dist_from_focus
