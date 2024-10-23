import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import vtk

from openlifu.util.units import getunitconversion


@dataclass
class Point:
    id: str = "point"
    name: str = "Point"
    color: Any = (1.0, 0.0, 0.0)
    radius: float = 1. # mm
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # mm
    dims: Tuple[str, str, str] = ("x","y","z")
    units: str = "mm"

    def __post_init__(self):
        if len(self.position) != len(self.dims):
            raise ValueError("Position and dims must have same length.")
        self.position = np.array(self.position).reshape(3)

    def copy(self):
        return copy.deepcopy(self)

    def get_position(self, dim=None, units: Optional[str] =None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        if dim is None:
            return self.position*scl
        else:
            return self.position[self.dims.index(dim)]*scl

    def get_matrix(self, origin: np.ndarray = np.eye(4), center_on_point: bool = False, local: bool = False):
        pos = np.dot(np.linalg.inv(origin), np.append(self.position, 1.0))[:3]
        if center_on_point:
            center = pos
        else:
            center = np.zeros(3)
        zvec = np.array([0., 0., 1.])
        if np.linalg.norm(pos) != 0:
            zvec = pos / np.linalg.norm(pos)
        az = -np.arctan2(zvec[0], zvec[2])
        xvec = np.array([np.cos(az), 0.0, np.sin(az)])
        yvec = np.cross(zvec, xvec)
        m = np.array([[xvec[0], yvec[0], zvec[0], center[0]],
                      [xvec[1], yvec[1], zvec[1], center[1]],
                      [xvec[2], yvec[2], zvec[2], center[2]],
                      [0.0, 0.0, 0.0, 1.0]])
        if not local:
            m = np.dot(origin, m)
        return m

    def get_polydata(self, transform: np.ndarray = np.eye(4), units=None):
        units = self.units if units is None else units
        colors =  vtk.vtkNamedColors()
        # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        scl = getunitconversion(self.units, units)
        pos = np.dot(transform, np.append(self.position*scl, 1.0))[:3]
        sphereSource.SetCenter(*pos)
        sphereSource.SetRadius(self.radius*scl)
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)
        return sphereSource

    def get_actor(self, transform: np.ndarray = np.eye(4), units=None):
        polydata = self.get_polydata(transform=transform, units=units)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(polydata.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.color)
        return actor

    def rescale(self, units: str):
        scl = getunitconversion(self.units, units)
        self.position = self.position * scl
        self.radius = self.radius * scl
        self.units = units

    def transform(self,
                  matrix: np.ndarray,
                  units: Optional[str] = None,
                  new_dims: Optional[Tuple[str, str, str]]=None):
        if units is not None:
            self.rescale(units)
        self.position = np.dot(matrix, np.append(self.position, 1.0))[:3]
        if new_dims is not None:
            self.dims = new_dims

    def to_dict(self):
        return {"id": self.id,
                "name": self.name,
                "color": self.color,
                "radius": self.radius,
                "position": self.position.tolist(),
                "dims": self.dims,
                "units": self.units}

    @staticmethod
    def from_dict(point_data:Dict):
        """Create a Point object from a dictionary."""
        if "color" in point_data:
            if len(point_data["color"]) != 3:
                raise ValueError(f"Color should have three components; got {point_data['color']}.")
            point_data["color"] = tuple(float(point_data["color"][i]) for i in range(3))
        if "radius" in point_data:
            point_data["radius"] = float(point_data["radius"])
        if "position" in point_data:
            point_data["position"] = np.array(point_data["position"])
        if "dims" in point_data:
            point_data["dims"] = tuple(point_data["dims"])
        return Point(**point_data)

    @staticmethod
    def from_json(json_string : str) -> "Point":
        """Load a Point from a json string"""
        return Point.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Point to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Point object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)
