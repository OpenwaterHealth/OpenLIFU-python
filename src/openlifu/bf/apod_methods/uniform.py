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
class Uniform(ApodizationMethod):
    value: Annotated[float, OpenLIFUFieldData("Value", "Uniform apodization value between 0 and 1.")] = 1.0
    """Uniform apodization value between 0 and 1."""

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        return np.full(arr.numelements(), self.value)
