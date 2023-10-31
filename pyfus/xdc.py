import numpy as np
import pandas as pd
from pyfus.util.units import getunitconversion
from dataclasses import dataclass, field
from collections.abc import Iterable 
from typing import List, Dict, Any, Tuple
import vtk
import logging
import json
import copy

@dataclass
class Element:
    index: int = 0
    x: float = 0
    y: float = 0
    z: float = 0
    az: float = 0
    el: float = 0
    roll: float = 0
    w: float = field(repr=False, default = 1)
    l: float = field(repr=False, default = 1)
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

    def rescale(self, units, inplace=False):
        if inplace:
            el = self
        else:
            el = self.copy()
        if el.units != units:
            scl = getunitconversion(el.units, units)
            el.x *= scl
            el.y *= scl
            el.z *= scl
            el.w *= scl
            el.l *= scl
            el.units = units
        if not inplace:
            return el

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

    def get_matrix(self):
        Raz = np.array([[np.cos(self.az), 0, np.sin(self.az)],
                        [0, 1, 0],
                        [-np.sin(self.az), 0, np.cos(self.az)]])
        Rel = np.array([[1, 0, 0],
                        [0, np.cos(self.el), -np.sin(self.el)],
                        [0, np.sin(self.el), np.cos(self.el)]])
        Rroll = np.array([[np.cos(self.roll), -np.sin(self.roll), 0],
                            [np.sin(self.roll), np.cos(self.roll), 0],
                            [0, 0, 1]])
        
        m = np.concatenate((np.dot(Raz, np.dot(Rel,Rroll)), [[self.x], [self.y], [self.z]]), axis=1)
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
        prev_units = self.units
        self.rescale(units, inplace=True)
        pos = np.array([self.x, self.y, self.z, 1])
        m = self.get_matrix()
        gm = np.dot(matrix, m)
        gpos = np.dot(matrix, pos)
        vec = point - gpos[:3]
        dist = np.linalg.norm(vec, 2)
        self.rescale(prev_units, inplace=True)
        return dist

    def angle_to_point(self, point, units="rad", matrix=np.eye(4)):
        m = self.get_matrix()
        gm = np.dot(matrix, m)
        v1 = point - gm[:3, 3]
        v2 = gm[:3, 2]
        v1 = v1 / np.linalg.norm(v1, 2)
        v2 = v2 / np.linalg.norm(v2, 2)
        vcross = np.cross(v1, v2)
        theta = np.arcsin(np.linalg.norm(vcross, 2))
        if units == "deg":
            theta = np.degrees(theta)
        return theta
    
    def set_matrix(self, matrix, units=None):
        if units is not None:
            self.rescale(units, inplace=True)
        x, y, z, az, el, roll = matrix2xyz(matrix)
        self.x = x
        self.y = y
        self.z = z
        self.az = az
        self.el = el
        self.roll = roll

    @staticmethod
    def from_dict(d):
        if isinstance(d, dict):
            return [Element(**d)]
        else:
            return [Element(**di) for di in d]

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
class Transducer:
    id: str = "transducer"
    name: str = ""
    elements: Tuple[Element] = ()
    frequency: float = 400.6e3
    units: str = "m"
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    attrs: Dict[str, Any] = field(default_factory= lambda: {})

    def __post_init__(self):
        logging.info("Initializing transducer array")
        if self.name == "":
            self.name = self.id
        self.matrix = np.array(self.matrix, dtype=np.float64)
        for element in self.elements:
            element.rescale(self.units, inplace=True)
    
    def calc_output(self, input_signal, dt, delays: np.ndarray = None, apod: np.ndarray = None):
        if delays is None:
            delays = np.zeros(self.numelements())
        if apod is None:
            apod = np.ones(self.numelements())
        outputs = [np.concatenate([np.zeros(int(delay/dt)), a*element.calc_output(input_signal, dt)],axis=0) for element, delay, a, in zip(self.elements, delays, apod)]
        max_len = max([len(o) for o in outputs])
        output_signal = np.zeros([self.numelements(), max_len])
        for i, o in enumerate(outputs):
            output_signal[i, :len(o)] = o
        return output_signal
    
    def copy(self):
        return copy.deepcopy(self)

    def draw(self, 
             units=None, 
             transform=True, 
             facecolor=[0,1,1], 
             facealpha=0.5):
        units = self.units if units is None else units
        prev_units = self.units
        self.rescale(units=units, inplace=True)
        actor = self.get_actor(transform=transform, facecolor=facecolor, facealpha=facealpha)
        self.rescale(units=prev_units, inplace=True)
        renderWindow = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)
        renderWindow.Render()
        renderWindowInteractor.Start()

    def get_actor(self, transform=False, facecolor=[0,1,1], facealpha=0.5):
        polydata = self.get_polydata(transform=transform, facecolor=facecolor, facealpha=facealpha)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def get_polydata(self, transform=False, facecolor=[0,1,1], facealpha=0.5):
        N = self.numelements()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4*N)
        cell_array = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4)
        color = (np.array([*facecolor, facealpha])*255).astype(np.uint8)
        point_index = 0
        if transform:
            matrix = self.get_matrix()
        else:
            matrix = np.eye(4)
        for el in self.elements:
            corners = el.get_corners(matrix=matrix)
            rect = vtk.vtkQuad()
            point_ids = rect.GetPointIds()
            for i in range(4):
                points.SetPoint(point_index, corners[:,i])
                point_ids.SetId(i, point_index)
                colors.InsertNextTuple4(*color)
                point_index += 1
            cell_array.InsertNextCell(rect)
        polydata = vtk.vtkPolyData()
        polydata.SetPolys(cell_array)
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        return polydata

    def get_area(self, units=None):
        units = self.units if units is None else units
        widths, lengths = zip(*[element.get_size(units=units) for element in self.elements])
        return sum(w * l for w, l in zip(widths, lengths))

    def get_corners(self, transform=True, units=None):
        units = self.units if units is None else units
        prev_units = self.units
        self.rescale(units=units, inplace=True)
        if transform:
            matrix = self.matrix
        else:
            matrix = np.eye(4)
        self.rescale(units=prev_units, inplace=True)
        return [element.get_corners(units=units, matrix=matrix) for element in self.elements]
  
    def get_positions(self, transform=True, units=None):
        units = self.units if units is None else units
        if transform:
            matrix = self.get_matrix(units=units)
        else:
            matrix = np.eye(4)
        positions = [element.get_position(units=units, matrix=matrix) for element in self.elements]
        return np.array(positions)

    def get_matrix(self, units=None):
        units = self.units if units is None else units
        matrix = self.matrix.copy()
        matrix[0:3, 3] *= getunitconversion(self.units, units)
        return matrix

    def get_unit_vectors(self, transform=True, scale=1, units=None):
        units = self.units if units is None else units
        prev_units = self.units
        self.rescale(units=units, inplace=True)
        unit_vectors = [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1]]
        ]
        unit_vectors = [[v * scale for v in uv] for uv in unit_vectors]
        if transform:
            matrix = self.matrix
            unit_vectors = [(np.dot(matrix, np.concatenate([np.array(uv).T, np.ones([1,2])], 0))[:3,:].T).tolist() for uv in unit_vectors]
        self.rescale(units=prev_units, inplace=True)
        return [np.array(uv).transpose() for uv in unit_vectors]

    def merge(self, list_of_transducers, inplace=False):
        if inplace:
            merged_array = self
        else:
            merged_array = self.copy()
        ref_matrix = merged_array.get_matrix()
        for arr in list_of_transducers:
            xform_array = arr.transform(np.dot(np.linalg.inv(arr.get_matrix()),ref_matrix), transform_elements=True)
            merged_array.elements += xform_array.elements
        if not inplace:
            return merged_array
        
    def numelements(self):
        return len(self.elements)

    def rescale(self, units, inplace=False):
        if inplace:
            array = self
        else:
            array = self.copy()
        if array.units != units:
            for element in array.elements:
                element.rescale(units, inplace=True)
            scl = getunitconversion(array.units, units)
            array.matrix[0:3, 3] *= scl
            array.units = units
        if not inplace:
            return array

    def to_dict(self):
        d = self.__dict__.copy()
        d["elements"] = [element.__dict__ for element in d["elements"]]
        return d

    def transform(self, matrix, units=None, transform_elements: bool=False, inplace: bool=False):
        if inplace:
            trans = self
        else:
            trans = self.copy()
        if units is not None:
            trans.rescale(units, inplace=True)
        if transform_elements:
            for el in trans.elements:
                el.set_matrix(np.dot(np.linalg.inv(matrix), el.get_matrix()))
            else:
                trans.matrix = np.dot(trans.matrix, np.linalg.inv(matrix))
        if not inplace:
            return trans

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return Transducer.from_dict(data)

    @staticmethod
    def from_dict(d, **kwargs):
        d = d.copy()
        d["elements"] = Element.from_dict(d["elements"])
        return Transducer(**d, **kwargs)

    @staticmethod
    def gen_matrix_array(nx=2, ny=2, pitch=1, kerf=0, units="mm", impulse_response=1, impulse_dt=1, id='array', name='Array', attrs={}):
        N = nx * ny
        xpos = [(i - nx // 2) * pitch for i in range(nx)]
        ypos = [(i - ny // 2) * pitch for i in range(ny)]
        elements = []
        for i in range(N):
            x = xpos[i % nx]
            y = ypos[i // nx]
            elements.append(Element(
                x=x,
                y=y,
                z=0,
                az=0,
                el=0,
                roll=0,
                w=pitch - kerf,
                l=pitch - kerf,
                impulse_response=impulse_response,
                impulse_dt=impulse_dt,
                units=units
            ))
        return Transducer(elements=elements, id=id, name=name, attrs=attrs)
    


