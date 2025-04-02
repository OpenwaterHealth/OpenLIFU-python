from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Tuple

import numpy as np
import vtk

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion


# === Tools to work with points ===
@dataclass
class Point:
    position: Annotated[np.ndarray, OpenLIFUFieldData("Position", "3D position of the point in the provided units")] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # mm
    """3D position of the point in the provided units"""

    id: Annotated[str, OpenLIFUFieldData("Point ID", "Unique identifier for the point")] = "point"
    """Unique identifier for the point"""

    name: Annotated[str, OpenLIFUFieldData("Point name", "Name of the point")] = "Point"
    """Name of the point"""

    color: Annotated[Any, OpenLIFUFieldData("Color (RGB)", "RGB color of the point")] = (1.0, 0.0, 0.0)
    """RGB color of the point"""

    radius: Annotated[float, OpenLIFUFieldData("Radius", "Radius for rendering the point in the provided units")] = 1.0  # mm
    """Radius for rendering the point in the provided units"""

    dims: Annotated[Tuple[str, str, str], OpenLIFUFieldData("Dimensions", "Names of the axes of the coordinate system being used")] = ("x", "y", "z")
    """Names of the axes of the coordinate system being used"""

    units: Annotated[str, OpenLIFUFieldData("Units", "Units for the point")] = "mm"
    """Units for the point"""

    def __post_init__(self):
        if len(self.position) != len(self.dims):
            raise ValueError("Position and dims must have same length.")
        self.position = np.array(self.position).reshape(3)

    def copy(self):
        return copy.deepcopy(self)

    def get_position(self, dim=None, units: str | None =None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        if dim is None:
            return self.position*scl
        else:
            return self.position[self.dims.index(dim)]*scl

    def get_matrix(self, origin: np.ndarray = np.eye(4), center_on_point: bool = True, local: bool = False):
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
                  units: str | None = None,
                  new_dims: Tuple[str, str, str] | None=None):
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
    def from_json(json_string : str) -> Point:
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


@dataclass
class ArrayTransform(DictMixin):
    """An affine transform with a unit string, often intended to represent how a transducer array is positioned in space."""

    matrix: Annotated[np.ndarray, OpenLIFUFieldData("Affine matrix", "4x4 affine transform matrix")]
    """4x4 affine transform matrix"""

    units: Annotated[str, OpenLIFUFieldData("Units", "The units of the space on which to apply the transform matrix , e.g. 'mm' (In order to apply the transform to points, first represent the points in these units.)")]
    """The units of the space on which to apply the transform matrix , e.g. "mm"
    (In order to apply the transform to points, first represent the points in these units.)
    """

# === Tools to work with spherical coordinate systems ===

def cartesian_to_spherical(x:float,y:float,z:float) -> Tuple[float, float, float]:
    """Convert cartesian coordinates to spherical coordinates

    Args: x, y, z are cartesian coordinates
    Returns: r, theta, phi, where
        r is the radial spherical coordinate, a nonnegative float.
        theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle.
            theta is in the range [0,pi].
        phi is the azimuthal spherical coordinate, in the range [-pi,pi]

    Angles are in radians.
    """
    return (np.sqrt(x**2+y**2+z**2), np.arctan2(np.sqrt(x**2+y**2),z), np.arctan2(y,x))

def spherical_to_cartesian(r: float, th:float, ph:float) -> Tuple[float, float, float]:
    """Convert spherical coordinates to cartesian coordinates

    Args:
        r: the radial spherical coordinate
        th: the polar spherical coordinate theta, aka the angle off the z-axis, aka the non-azimuthal spherical angle
        ph: the azimuthal spherical coordinate phi
    Returns the cartesian coordinates x,y,z

    Angles are in radians.
    """
    return (r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th))

def cartesian_to_spherical_vectorized(p:np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates to spherical coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point cartesian coordinates x,y,z.
    Returns: An array of shape (...,3), where the last axis describes point spherical coordinates r, theta, phi, where
        r is the radial spherical coordinate, a nonnegative float.
        theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle.
            theta is in the range [0,pi].
        phi is the azimuthal spherical coordinate, in the range [-pi,pi]

    Angles are in radians.
    """
    return np.stack([
        np.sqrt((p**2).sum(axis=-1)),
        np.arctan2(np.sqrt((p[...,0:2]**2).sum(axis=-1)),p[...,2]),
        np.arctan2(p[...,1],p[...,0]),
    ], axis=-1)

def spherical_to_cartesian_vectorized(p:np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point spherical coordinates r, theta, phi, where:
            r is the radial spherical coordinate
            theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle
            phi is the azimuthal spherical coordinate
    Returns the cartesian coordinates x,y,z

    Angles are in radians.
    """
    return np.stack([
        p[...,0]*np.sin(p[...,1])*np.cos(p[...,2]),
        p[...,0]*np.sin(p[...,1])*np.sin(p[...,2]),
        p[...,0]*np.cos(p[...,1]),
    ], axis=-1)

def spherical_coordinate_basis(th:float, phi:float) -> np.ndarray:
    """Return normalized spherical coordinate basis at a location with spherical polar and azimuthal coordinates (th, phi).
    The coordinate basis is returned as an array `basis` of shape (3,3), where the rows are the basis vectors,
    in the order r, theta, phi. So `basis[0], basis[1], basis[2]` are the vectors $\\hat{r}$, $\\hat{\\theta}$, $\\hat{\\phi}$.
    Angles are assumed to be provided in radians."""
    return np.array([
        [np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)],
        [np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0],
    ])

# === General coordinate transformation utiltiies ===

def create_standoff_transform(z_offset:float, dzdy:float) -> np.ndarray:
    """Create a standoff transform based on a z_offset and a dzdy value.

    A "standoff transform" applies a displacement in transducer space that moves a transducer to where it would
    be situated with the standoff in place. The idea is that if you start with a transform that places a transducer
    directly against skin, then pre-composing that transform by a "standoff transform" serves to nudge the transducer
    such that there is space for the standoff to be between it and the skin.

    This function assumes that the standoff is laterally symmetric, has some thickness, and can raise the bottom of
    the transducer a bit more than the top. The `z_offset` is the thickness in the middle of the standoff,
    while the `dzdy` is the elevational slope.

    Args:
        z_offset: Thickness in the middle of the standoff
        dzdy: Slope of the standoff, as axial displacement per unit elevational displacement. A positive number
            here means that the bottom of the transducer is raised a little bit more than the top.

    Returns a 4x4 matrix representing a rigid transform in whatever units z_offset was provided in.
    """
    angle = np.arctan(dzdy)
    return np.array([
        [1,0,0,0],
        [0,np.cos(angle),-np.sin(angle),0],
        [0,np.sin(angle),np.cos(angle),-z_offset],
        [0,0,0,1],
    ], dtype=float)
