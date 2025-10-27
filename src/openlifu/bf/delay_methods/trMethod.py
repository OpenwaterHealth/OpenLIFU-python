from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.xdc import Transducer
from openlifu.sim.time_reversal import TimeReversal

class TRDelay(DelayMethod):
    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Speed of sound in the medium (m/s)")] = 1480.0

    def __post_init__(self):
        if not isinstance(self.c0, (int, float)):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)
    
    def calc_delays(self, arr, target, params, transform = None, medium = None):
        if params is None:
            c = self.c0
        else:
            
            tr = TimeReversal(kgrid,medium,sensor)
            c = medium["sound_speed"]
        return super().calc_delays(arr, target, params, transform)
        