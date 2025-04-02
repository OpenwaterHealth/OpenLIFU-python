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
class Uniform(ApodizationMethod):
    value: Annotated[float, OpenLIFUFieldData("Value", None)] = 1.0
    """TODO: Add description"""

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        return np.full(arr.numelements(), self.value)
