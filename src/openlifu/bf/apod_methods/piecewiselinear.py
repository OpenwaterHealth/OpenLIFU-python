from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.xdc import Transducer


@dataclass
class PiecewiseLinear(ApodizationMethod):
    zero_angle: Annotated[float, OpenLIFUFieldData("Zero Apodization Angle", "Angle at and beyond which the piecewise linear apodization is 0%")] = 90.0
    """Angle at and beyond which the piecewise linear apodization is 0%"""

    rolloff_angle: Annotated[float, OpenLIFUFieldData("Rolloff start angle", "Angle below which the piecewise linear apodization is 100%")] = 45.0
    """Angle below which the piecewise linear apodization is 100%"""

    units: Annotated[str, OpenLIFUFieldData("Angle units", "Angle units")] = "deg"
    """Angle units"""

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        f = ((self.zero_angle - angles) / (self.zero_angle - self.rolloff_angle))
        apod = np.maximum(0, np.minimum(1, f))
        return apod
