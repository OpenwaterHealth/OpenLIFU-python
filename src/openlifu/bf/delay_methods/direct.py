from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class Direct(DelayMethod):
    c0: float = 1480.0
    def calc_delays(self, arr: Transducer, target: Point, params: Optional[xa.Dataset]=None, transform: bool = True):
        if params is None:
            c = self.c0
        else:
            c = params['sound_speed'].attrs['ref_value']
        target_pos = target.get_position(units="m")
        matrix = arr.get_matrix(units="m") if transform else np.eye(4)
        dists = np.array([el.distance_to_point(target_pos, units="m", matrix=matrix) for el in arr.elements])
        tof = dists / c
        delays = max(tof) - tof
        return delays
