from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class PiecewiseLinear(ApodizationMethod):
    zero_angle: float = 90.0
    rolloff_angle: float = 45.0
    units: str = "deg"
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:Optional[np.ndarray]=None):
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        f = ((self.zero_angle - angles) / (self.zero_angle - self.rolloff_angle))
        apod = np.maximum(0, np.minimum(1, f))
        return apod
