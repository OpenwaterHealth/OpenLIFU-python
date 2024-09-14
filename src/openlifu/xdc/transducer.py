
import numpy as np
from openlifu.util.units import getunitconversion
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import vtk
import logging
import json
import copy
from .element import Element

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
            element.rescale(self.units)

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
             facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        actor = self.get_actor(units=units, transform=transform, facecolor=facecolor)
        renderWindow = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)
        renderWindow.Render()
        renderWindowInteractor.Start()

    def get_actor(self, units=None, transform=False, facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        polydata = self.get_polydata(units=units, transform=transform, facecolor=facecolor)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def get_polydata(self, units=None, transform=False, facecolor=None):
        units = self.units if units is None else units
        N = self.numelements()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4*N)
        cell_array = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4)
        facecolor = np.array(facecolor)
        is_color_el_none = np.vectorize(lambda color_el: color_el is None)
        if np.all(is_color_el_none(facecolor)):
            facecolors = np.tile((np.array(None)), (N, 1))
        elif facecolor.ndim == 1:
            facecolors = np.tile((np.array([*facecolor])*255).astype(np.uint8), (N, 1))
        else:
            facecolors = np.array([np.array([*fc])*255 for fc in facecolor]).astype(np.uint8)
        point_index = 0
        if transform:
            matrix = self.get_matrix(units=units)
        else:
            matrix = np.eye(4)
        for el, color in zip(self.elements, facecolors):
            corners = el.get_corners(matrix=matrix, units=units)
            rect = vtk.vtkQuad()
            point_ids = rect.GetPointIds()
            for i in range(4):
                points.SetPoint(point_index, corners[:,i])
                point_ids.SetId(i, point_index)
                if color[0] is not None:
                    colors.InsertNextTuple4(*color)
                point_index += 1
            cell_array.InsertNextCell(rect)
        polydata = vtk.vtkPolyData()
        polydata.SetPolys(cell_array)
        polydata.SetPoints(points)
        if not np.all(is_color_el_none(facecolor)):
            polydata.GetPointData().SetScalars(colors)
        return polydata

    def get_area(self, units=None):
        units = self.units if units is None else units
        widths, lengths = zip(*[element.get_size(units=units) for element in self.elements])
        return sum(w * l for w, l in zip(widths, lengths))

    def get_corners(self, transform=True, units=None):
        units = self.units if units is None else units
        if transform:
            matrix = self.get_matrix(units=units)
        else:
            matrix = np.eye(4)
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
        unit_vectors = [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1]]
        ]
        unit_vectors = [[v * scale for v in uv] for uv in unit_vectors]
        if transform:
            matrix = self.get_matrix(units=units)
            unit_vectors = [(np.dot(matrix, np.concatenate([np.array(uv).T, np.ones([1,2])], 0))[:3,:].T).tolist() for uv in unit_vectors]
        return [np.array(uv).transpose() for uv in unit_vectors]

    @staticmethod
    def merge(list_of_transducers):
        merged_array = list_of_transducers[0].copy()
        ref_matrix = merged_array.get_matrix()
        for arr in list_of_transducers[1:]:
            xform_array = arr.copy()
            xform_array.transform(np.dot(np.linalg.inv(arr.get_matrix()),ref_matrix), transform_elements=True)
            merged_array.elements += xform_array.elements
        return merged_array

    def numelements(self):
        return len(self.elements)

    def rescale(self, units):
        if self.units != units:
            for element in self.elements:
                element.rescale(units)
            scl = getunitconversion(self.units, units)
            self.matrix[0:3, 3] *= scl
            self.units = units

    def to_dict(self):
        d = self.__dict__.copy()
        d["elements"] = [element.to_dict() for element in d["elements"]]
        d["matrix"] = d["matrix"].tolist()
        return d

    def to_file(self, filename):
        from openlifu.util.json import to_json
        to_json(self.to_dict(), filename)

    def transform(self, matrix, units=None, transform_elements: bool=False):
        if units is not None:
            self.rescale(units)
        if transform_elements:
            for el in self.elements:
                el.set_matrix(np.dot(np.linalg.inv(matrix), el.get_matrix()))
            else:
                self.matrix = np.dot(self.matrix, np.linalg.inv(matrix))

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return Transducer.from_dict(data)

    @staticmethod
    def from_dict(d, **kwargs):
        d = d.copy()
        d["elements"] = Element.from_dict(d["elements"])
        d["matrix"] = np.array(d["matrix"])
        return Transducer(**d, **kwargs)

    @staticmethod
    def from_json(json_string : str) -> "Transducer":
        """Load a Transducer from a json string"""
        return Transducer.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Transducer to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Transducer object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

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
