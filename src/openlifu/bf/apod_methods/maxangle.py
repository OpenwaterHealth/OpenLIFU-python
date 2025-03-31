from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.util.openlifu_annotations import OpenLIFUFieldData
from openlifu.xdc import Transducer


@dataclass
class MaxAngle(ApodizationMethod):
    max_angle: Annotated[float, OpenLIFUFieldData("Maximum angle", "TODO: Add description")] = 30.0
    """TODO: Add description"""

    units: Annotated[str, OpenLIFUFieldData("Angle units", "TODO: Add description")] = "deg"
    """TODO: Add description"""

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        apod[angles <= self.max_angle] = 1
        return apod
