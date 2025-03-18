from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class Uniform(ApodizationMethod):
    value = 1
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        return np.full(arr.numelements(), self.value)
