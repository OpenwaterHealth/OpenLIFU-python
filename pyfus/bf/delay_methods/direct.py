from dataclasses import dataclass
import xarray as xa
import numpy as np
from pyfus import Transducer, Point
from pyfus.bf.delay_methods import DelayMethod
from typing import ClassVar

@dataclass
class Direct(DelayMethod):
    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: bool = True):
        c = params['sound_speed'].attrs['ref_value']
        target_pos = target.get_position(units="m")
        matrix = arr.get_matrix(units="m") if transform else np.eye(4)
        dists = np.array([el.distance_to_point(target_pos, units="m", matrix=matrix) for el in arr.elements])
        tof = dists / c
        delays = max(tof) - tof
        return delays
