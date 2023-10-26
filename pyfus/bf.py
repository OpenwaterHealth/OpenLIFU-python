import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pyfus import geo
from typing import Any, Tuple, Optional
from pyfus.util.units import getunitconversion
from pyfus.xdc import Transducer
import xarray as xa

@dataclass
class Pulse:
    frequency: float = 1.0 # Hz
    amplitude: float = 1.0 # Pa
    duration: float = 1.0 # s

    def calc_pulse(self, t: np.array):
        return self.amplitude * np.sin(2*np.pi*self.frequency*t)
    
    def calc_time(self, dt: float):
        return np.arange(0, self.duration, dt)
    
    def get_table(self):
        records = [{"Name": "Frequency", "Value": self.frequency, "Unit": "Hz"},
                   {"Name": "Amplitude", "Value": self.amplitude, "Unit": "Pa"},
                   {"Name": "Duration", "Value": self.duration, "Unit": "s"}]
        return pd.DataFrame.from_records(records)
    
    def to_dict(self):
        return {"frequency": self.frequency,
                "amplitude": self.amplitude,
                "duration": self.duration,
                "class": "Pulse"}
    
    @staticmethod
    def from_dict(d):
        return Pulse(frequency=d["frequency"], amplitude=d["amplitude"], duration=d["duration"])
    
@dataclass
class Sequence:
    pulse_interval: float = 1.0 # s
    pulse_count: int = 1
    pulse_train_interval: float = 1.0 # s
    pulse_train_count: int = 1

    def get_table(self):
        records = [{"Name": "Pulse Interval", "Value": self.pulse_interval, "Unit": "s"},
                   {"Name": "Pulse Count", "Value": self.pulse_count, "Unit": ""},
                   {"Name": "Pulse Train Interval", "Value": self.pulse_train_interval, "Unit": "s"},
                   {"Name": "Pulse Train Count", "Value": self.pulse_train_count, "Unit": ""}]
        return pd.DataFrame.from_records(records)
    
    @staticmethod
    def from_dict(d):
        return Sequence(pulse_interval=d["pulse_interval"], 
                        pulse_count=d["pulse_count"],
                        pulse_train_interval=d["pulse_train_interval"], 
                        pulse_train_count=d["pulse_train_count"])
    
@dataclass
class FocalPattern(ABC):
    target_pressure: float = 1.0 # Pa

    @abstractmethod
    def get_targets(self, target: geo.Point):
        pass

    @abstractmethod
    def num_foci(self):
        pass

    @staticmethod
    def from_dict(d):
        d = d.copy()
        class_constructor = globals()[d.pop("class")]
        return class_constructor(**d)

@dataclass
class SingleFocus(FocalPattern):
    def get_targets(self, target: geo.Point):
        return target

    def num_foci(self):
        return 1
    
@dataclass
class RadialPattern(FocalPattern):
    center: bool = True
    num_spokes: int = 4
    spoke_radius: float = 1.0 # mm   
    units: str = "mm"

    def get_targets(self, target: geo.Point):
        if self.center:
            targets = [target.copy()]
            targets[0].id = f"{target.id}_center"
            targets[0].id = f"{target.id} (Center)"
        else:    
            targets = []
        m = target.get_matrix(center_on_point=True)
        for i in range(self.num_spokes):
            theta = 2*np.pi*i/self.num_spokes
            local_position = self.spoke_radius * np.array([np.cos(theta), np.sin(theta), 0.0])
            position = np.dot(m, np.append(local_position, 1.0))[:3]
            spoke = geo.Point(id=f"{target.id}_{np.rad2deg(theta):.0f}deg",
                              name=f"{target.name} ({np.rad2deg(theta):.0f}°)",
                              position=position,
                              units=self.units,
                              radius=target.radius)
            targets.append(spoke)
        return targets
    
    def num_foci(self):
        return int(self.center) + self.num_spokes

@dataclass
class SimulationGrid:
    dims: Tuple[str, str, str] = ("lat", "ele", "ax")
    names: Tuple[str, str, str] = ("Lateral", "Elevation", "Axial")
    spacing: float = 1.0
    units: str = "mm"
    x_extent: Tuple[float, float] = (-30., 30.)
    y_extent: Tuple[float, float] = (-30., 30.)
    z_extent: Tuple[float, float] = (-4., 60.)
    dt: float = 0.
    t_end: float = 0.
    options: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        if len(self.dims) != 3:
            raise ValueError("dims must have length 3.")
        if len(self.names) != 3:
            raise ValueError("names must have length 3.")
        if len(self.x_extent) != 2:
            raise ValueError("x_extent must have length 2.")
        if len(self.y_extent) != 2:
            raise ValueError("y_extent must have length 2.")
        if len(self.z_extent) != 2:
            raise ValueError("z_extent must have length 2.")
        self.dims = tuple(self.dims)
        self.names = tuple(self.names)
        self.x_extent = tuple(self.x_extent)
        self.y_extent = tuple(self.y_extent)
        self.z_extent = tuple(self.z_extent)

    def get_coords(self, dims=None, units: Optional[str] = None):
        raise NotImplementedError
    
    def get_corners(self, id: str = "corners", units: Optional[str] = None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        xyz = np.array(np.meshgrid(self.x_extent, self.y_extent, self.z_extent, indexing='ij'))
        corners = xyz.reshape(3,-1)
        return corners*scl
    
    def get_max_distance(self, arr: Transducer, units: Optional[str] = None):
        units = self.units if units is None else units
        corners = self.get_corners(units=units)
        distances = np.array([[el.distance_to_point(corner) for corner in corners.T] for el in arr.rescale(units).elements])
        max_distance = np.max(distances)
        return max_distance
    
    def transform_scene(self, scene, id: Optional[str] = None, name: Optional[str] = None, units: Optional[str] = None):
        raise NotImplementedError
