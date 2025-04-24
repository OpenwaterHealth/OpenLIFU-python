from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion


def matrix2xyz(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    az = np.arctan2(matrix[0, 2], matrix[2, 2])
    el = -np.arctan2(matrix[1, 2], np.sqrt(matrix[2, 2]**2 + matrix[0, 2]**2))
    Raz = np.array([[np.cos(az), 0, np.sin(az)],
                    [0, 1, 0],
                    [-np.sin(az), 0, np.cos(az)]])
    Rel = np.array([[1, 0, 0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el), np.cos(el)]])
    Razel = np.dot(Raz, Rel)
    xv = matrix[:3, 0]
    xyp = np.dot(xv, Razel[:3,1])
    xxp = np.dot(xv, Razel[:3,0])
    roll = np.arctan2(xyp, xxp)
    return x, y, z, az, el, roll

@dataclass
class Element:
    index: Annotated[int, OpenLIFUFieldData("Element index", "Element index")] = 0
    """Element index to identify the element in the array."""

    x: Annotated[float, OpenLIFUFieldData("X position", "X position of the element")] = 0
    """X position of the element."""

    y: Annotated[float, OpenLIFUFieldData("Y position", "Y position of the element")] = 0
    """Y position of the element."""

    z: Annotated[float, OpenLIFUFieldData("Z position", "Z position of the element")] = 0
    """Z position of the element."""

    az: Annotated[float, OpenLIFUFieldData("Azimuth angle (rad)", "Azimuth angle of the element")] = 0
    """Azimuth angle of the element, or rotation about the y-axis (rad)"""

    el: Annotated[float, OpenLIFUFieldData("Elevation angle (rad)", "Elevation angle of the element")] = 0
    """Elevation angle of the element, or rotation about the x'-axis (rad)"""

    roll: Annotated[float, OpenLIFUFieldData("Roll angle (rad)", "Roll angle of the element")] = 0
    """Roll angle of the element, or rotation about the z''-axis (rad)"""

    w: Annotated[float, OpenLIFUFieldData("Width", "Width of the element in the x dimension")] = 1
    """Width of the element in the x dimension"""

    l: Annotated[float, OpenLIFUFieldData("Length", "Length of the element in the y dimension")] = 1
    """Length of the element in the y dimension"""

    impulse_response: Annotated[np.ndarray, OpenLIFUFieldData("Impulse response", "Impulse response of the element")] = field(repr=False, default_factory=lambda: np.array([1]))
    """Impulse response of the element, can be a single value or an array of values. If an array, `impulse_dt` must be set to the time step of the impulse response. Is convolved with the input signal."""

    impulse_dt: Annotated[float, OpenLIFUFieldData("Impulse response timestep", """Impulse response timestep""")] = field(repr=False, default=1)
    """Impulse response timestep. If `impulse_response` is an array, this is the time step of the impulse response."""

    pin: Annotated[int, OpenLIFUFieldData("Pin", "Channel pin to which the element is connected")] = -1
    """Channel pin to which the element is connected. 1-(64*number of modules)."""

    units: Annotated[str, OpenLIFUFieldData("Units", "Spatial units")] = "mm"
    """Spatial units of the element specification."""

    def __post_init__(self):
        if isinstance(self.impulse_response, Iterable):
            self.impulse_response = np.array(self.impulse_response, dtype=np.float64)
        else:
            self.impulse_response = np.array([self.impulse_response], dtype=np.float64)

    def calc_output(self, input_signal, dt):
        if len(self.impulse_response) == 1:
            return input_signal * self.impulse_response
        else:
            impulse = self.interp_impulse_response(dt)
            return np.convolve(input_signal, impulse, mode='full')

    def copy(self):
        return copy.deepcopy(self)

    def rescale(self, units):
        if self.units != units:
            scl = getunitconversion(self.units, units)
            self.x *= scl
            self.y *= scl
            self.z *= scl
            self.w *= scl
            self.l *= scl
            self.units = units

    def get_position(self, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        pos = np.array([self.x, self.y, self.z]) * scl
        pos = np.append(pos, 1)
        pos = np.dot(matrix, pos)
        return pos[:3]

    def get_size(self, units=None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        ele_width = self.w * scl
        ele_length = self.l * scl
        return ele_width, ele_length

    def get_area(self, units=None):
        units = self.units if units is None else units
        ele_width, ele_length = self.get_size(units)
        return ele_width * ele_length

    def get_corners(self, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        rect = np.array([np.array([-1, -1.,  1,  1]) * 0.5 * self.w,
                            np.array([-1,  1,  1, -1]) * 0.5 * self.l,
                            np.zeros(4) ,
                            np.ones(4)])
        xyz = np.dot(self.get_matrix(), rect)
        xyz1 = np.dot(matrix, xyz)
        corner = []
        for j in range(3):
            corner.append(xyz1[j, :] * scl)
        return np.array(corner            )

    def get_matrix(self, units=None):
        units = self.units if units is None else units
        Raz = np.array([[np.cos(self.az), 0, np.sin(self.az)],
                        [0, 1, 0],
                        [-np.sin(self.az), 0, np.cos(self.az)]])
        Rel = np.array([[1, 0, 0],
                        [0, np.cos(self.el), -np.sin(self.el)],
                        [0, np.sin(self.el), np.cos(self.el)]])
        Rroll = np.array([[np.cos(self.roll), -np.sin(self.roll), 0],
                            [np.sin(self.roll), np.cos(self.roll), 0],
                            [0, 0, 1]])
        pos = self.get_position(units=units)
        m = np.concatenate((np.dot(Raz, np.dot(Rel,Rroll)), pos.reshape([3,1])), axis=1)
        m = np.concatenate((m, [[0, 0, 0, 1]]), axis=0)
        return m

    def get_angle(self, units="rad"):
        # Return angles about the x, y', and z'' axes (el, az, roll)
        if units == "rad":
            el = self.el
            az = self.az
            roll = self.roll
        elif units == "deg":
            el = np.degrees(self.el)
            az = np.degrees(self.az)
            roll = np.degrees(self.roll)
        return el, az, roll

    def interp_impulse_response(self, dt=None):
        if dt is None:
            dt = self.impulse_dt
        n0 = len(self.impulse_response)
        if n0 == 1:
            impulse_response= self.impulse_response
        else:
            t0 = self.impulse_dt * np.arange(n0)
            t1 = np.arange(0, t0[-1] + dt, dt)
            impulse_response = np.interp(t1, t0, self.impulse_response)
        impulse_t = np.arange(len(impulse_response)) * dt
        impulse_t = impulse_t - np.mean(impulse_t)
        return impulse_response, impulse_t

    def distance_to_point(self, point, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        pos = np.concatenate([self.get_position(units=units), [1]])
        m = self.get_matrix(units=units)
        gpos = np.dot(matrix, pos)
        vec = point - gpos[:3]
        dist = np.linalg.norm(vec, 2)
        return dist

    def angle_to_point(self, point, units=None, return_as="rad", matrix=np.eye(4)):
        units = self.units if units is None else units
        m = self.get_matrix(units=units)
        gm = np.dot(matrix, m)
        v1 = point - gm[:3, 3]
        v2 = gm[:3, 2]
        v1 = v1 / np.linalg.norm(v1, 2)
        v2 = v2 / np.linalg.norm(v2, 2)
        vcross = np.cross(v1, v2)
        theta = np.arcsin(np.linalg.norm(vcross, 2))
        if return_as == "deg":
            theta = np.degrees(theta)
        return theta

    def set_matrix(self, matrix, units=None):
        if units is not None:
            self.rescale(units)
        x, y, z, az, el, roll = matrix2xyz(matrix)
        self.x = x
        self.y = y
        self.z = z
        self.az = az
        self.el = el
        self.roll = roll

    def to_dict(self):
        return {"index": self.index,
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "az": self.az,
                "el": self.el,
                "roll": self.roll,
                "w": self.w,
                "l": self.l,
                "impulse_response": self.impulse_response.tolist(),
                "impulse_dt": self.impulse_dt,
                "pin": self.pin,
                "units": self.units}

    @staticmethod
    def from_dict(d):
        return Element(**d)
