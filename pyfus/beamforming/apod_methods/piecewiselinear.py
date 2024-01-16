from dataclasses import dataclass
import xarray as xa
import numpy as np
from pyfus import Transducer, Point
from pyfus.beamforming.apod_methods import ApodizationMethod

@dataclass
class PiecewiseLinear(ApodizationMethod):
    zero_angle: float = 90.0
    rolloff_angle: float = 45.0
    units: str = "deg"
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform: bool = True):
        target_pos = target.get_position(units="m")
        matrix = arr.get_matrix(units="m") if transform else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        f = ((self.zero_angle - angles) / (self.zero_angle - self.rolloff_angle))
        apod = np.maximum(0, np.minimum(1, f))
        return apod