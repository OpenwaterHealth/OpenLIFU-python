from enum import Enum
from typing import Any, Dict

import numpy as np

from openlifu.geo import Point

CenterOnOpts = Enum('CenterOnOpts', ['focus', 'origin'])


def get_focus_matrix(focus: Point, options: Dict[str, Any]):
    """
    Get transformation matrix for a focus point.

    Returns the transformation matrix for the focus point.
    The transformation matrix is a 4x4 matrix that transforms points
    in the coordinate system of the chosen origin to the global coordinate system.

    Args:
        focus : fus.Point object
            The focus point to be used as reference.
        options: dictionary
            Additional parameters:
                - 'units': str
                    Distance units. Default is "m".
                - 'center_on': CenterOnOpts
                    Whether to transform from focus or origin.
                    Choice between ["focus", "origin"]. Defaults to "focus".

    Returns:
        A 4x4 np.ndarray transformation matrix.
    """

    # Define default values for options if not provided
    if not hasattr(options, 'units') or options['units'] is None:
        units = "m"  #TODO: if fus.Axis is defined, use instead coords.get_units()
    else:
        units = options['units']
    if not hasattr(options, 'center_on') or options['center_on'] is None:
        center_on = CenterOnOpts.focus
    else:
        center_on = options['center_on']

    # Create rotation matrix and origin
    focus_rescaled = focus.rescale(units)
    zvec = focus.position
    zvec = zvec/np.linalg.norm(zvec)
    az = -np.arctan2(zvec[0], zvec[2])
    xvec = [np.cos(az), 0, np.sin(az)]
    yvec = np.cross(zvec, xvec)
    uv = np.column_stack([xvec, yvec, zvec])
    # Set origin point based on center_on option
    if center_on is CenterOnOpts.focus:
        origin = focus_rescaled
    elif center_on is CenterOnOpts.origin:
        origin = np.zeros(3, 1)

    # Create transformation matrix (in homogeneous coordinates)
    M = np.vstack((uv, origin, np.array([0., 0., 0., 1.])))

    return M
