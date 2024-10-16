import numpy as np
from xarray import DataArray

from openlifu.util.units import get_ndgrid_from_arr
from openlifu.geo import Point


def offset_grid(data_arr: DataArray, focus: Point):
    """
    Calculates the distance from the focus point for each point in the coordinate system.

    Distances are returned from a coordinate system rotated in azimuth,
    then elevation, so that the 'z' axis points at the focus.

    Args:
        data_arr: xarray.DataArray
        focus : fus.Point object
            The focus point to be used as reference.

    Returns:
        A list of distance arrays offsets 'dx', 'dy', 'dz' for each point in the coordinate system.
    """
    m = focus.get_matrix()

    # Create a grid of homogeneous points in the coordinate system
    xyz = get_ndgrid_from_arr(data_arr)
    XYZ = np.append(xyz, np.ones((*xyz.shape[:3], 1)), axis=-1)
    ogrid = XYZ @ np.linalg.inv(m).T[np.newaxis, np.newaxis, ...]

    return ogrid[..., :3]
