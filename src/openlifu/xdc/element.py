import numpy as np
from openlifu.util.units import getunitconversion
from dataclasses import dataclass, field
from collections.abc import Iterable
import copy

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
    index: int = 0
    x: float = 0
    y: float = 0
    z: float = 0
    az: float = 0
    el: float = 0
    roll: float = 0
    w: float = 1
    l: float = 1
    impulse_response: np.ndarray = field(repr=False, default_factory=lambda: np.array([1]))
    impulse_dt: float = field(repr=False, default = 1)
    pin: int = -1
    units: str = "mm"

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
        ele_width, ele_length = self.get_size({"units": units})
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
        if units == "rad":
            az = self.az
            el = self.el
            roll = self.roll
        elif units == "deg":
            az = np.degrees(self.az)
            el = np.degrees(self.el)
            roll = np.degrees(self.roll)
        return az, el, roll

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
        if isinstance(d, dict):
            return [Element(**d)]
        else:
            return [Element(**di) for di in d]
