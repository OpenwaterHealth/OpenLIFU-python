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

from openlifu.bf import apod_methods, focal_patterns, delay_methods
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan import Protocol
from openlifu.sim import SimSetup

from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

@dataclass
class TRDelay(DelayMethod):
    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Speed of sound in the medium (m/s)")] = 1480.0
    
    def __init__(self,kgrid,medium,sensor,pulse,sequence):
        self.kgrid = kgrid
        self.medium = medium
        self.sensor = sensor
        self.pulse = pulse
        self.sequence = sequence
        self.simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
        self.execution_options = SimulationExecutionOptions(is_gpu_simulation=True)


    def __post_init__(self):
        if not isinstance(self.c0, (int, float)):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)
    
    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset | None=None, transform:np.ndarray | None=None):
        if params is None:
            c = self.c0
        else:
            c = self.medium['sound_speed']
        

        focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
        apod_method = apod_methods.Uniform()
        delay_method = delay_methods.Direct()
        sim_setup = SimSetup(x_extent=(-55,55), y_extent=(-30,30), z_extent=(-4,150))
        protocol = Protocol(
            id='test_protocol',
            name='Test Protocol',
            pulse=self.pulse,
            sequence=self.sequence,
            focal_pattern=focal_pattern,
            apod_method=apod_method,
            sim_setup=sim_setup)

        _, sim_res, _ = protocol.calc_solution(
            target=target,
            transducer=arr,
            simulate=True,
            scale=True,
            use_gpu=True)
        
        t0 = sim_res['p_max']
        
        tr = TimeReversal(self.kgrid,self.medium,self.sensor,arr)
        result = tr()

        return delays

