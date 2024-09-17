from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class Uniform(ApodizationMethod):
    value = 1
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:Optional[np.ndarray]=None):
        return np.full(arr.numelements(), self.value)
