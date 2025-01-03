
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk

from openlifu.util.units import getunitconversion

from .element import Element


@dataclass
class Transducer:
    id: str = "transducer"
    name: str = ""
    elements: Tuple[Element] = ()
    frequency: float = 400.6e3
    units: str = "m"
    attrs: Dict[str, Any] = field(default_factory= dict)
    registration_surface_filename: Optional[str] = ""
    """Relative path to an open surface of the transducer to be used for registration"""
    transducer_body_filename: Optional[str] = ""
    """Relative path to the closed surface mesh for visualizing the transducer body"""

    def __post_init__(self):
        logging.info("Initializing transducer array")
        if self.name == "":
            self.name = self.id
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
             transform:Optional[np.ndarray]=None,
             units:Optional[str]=None,
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

    def get_actor(self, transform:Optional[np.ndarray]=None, units:Optional[str]=None, facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        polydata = self.get_polydata(units=units, transform=transform, facecolor=facecolor)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def get_polydata(self, transform:Optional[np.ndarray]=None, units:Optional[str]=None, facecolor=None):
        """Get a vtk polydata of the transducer. Optionally provide a transform, and units in which to interpret
        that transform. If a transform is provided with no units specified, it is assumed that the units
        are the same as those of the transducer itself. Optionally provide an RGBA color to set."""
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
        matrix = transform if transform is not None else np.eye(4)
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

    def get_corners(self, transform:Optional[np.ndarray]=None, units:Optional[str]=None):
        units = self.units if units is None else units
        matrix = transform if transform is not None else np.eye(4)
        return [element.get_corners(units=units, matrix=matrix) for element in self.elements]

    def get_positions(self, transform:Optional[np.ndarray]=None, units:Optional[str]=None):
        units = self.units if units is None else units
        matrix = transform if transform is not None else np.eye(4)
        positions = [element.get_position(units=units, matrix=matrix) for element in self.elements]
        return np.array(positions)

    def convert_transform(self, matrix:np.ndarray, units:str) -> np.ndarray:
        """Given a transform matrix in some units, convert it to this transducer's native units.

        Args:
            matrix: 4x4 affine transform matrix
            units: units of the coordinate space on which the provided transform matrix operates

        Returns: 4x4 affine transform matrix, now operating on a the transducer's native coordinate space
            (i.e. in the transducer's native units)
        """
        matrix = matrix.copy()
        matrix[0:3, 3] *= getunitconversion(units, self.units)
        return matrix

    @staticmethod
    def merge(list_of_transducers:"List[Transducer]") -> "Transducer":
        merged_array = list_of_transducers[0].copy()
        for arr in list_of_transducers[1:]:
            xform_array = arr.copy()
            merged_array.elements += xform_array.elements
        return merged_array

    def numelements(self):
        return len(self.elements)

    def rescale(self, units):
        if self.units != units:
            for element in self.elements:
                element.rescale(units)
            self.units = units

    def to_dict(self):
        d = self.__dict__.copy()
        d["elements"] = [element.to_dict() for element in d["elements"]]
        return d

    def to_file(self, filename):
        from openlifu.util.json import to_json
        to_json(self.to_dict(), filename)

    def transform(self, matrix, units=None):
        if units is not None:
            self.rescale(units)
        for el in self.elements:
            el.set_matrix(np.dot(np.linalg.inv(matrix), el.get_matrix()))

    @staticmethod
    def from_file(filename):
        with open(filename) as file:
            data = json.load(file)
        return Transducer.from_dict(data)

    @staticmethod
    def from_dict(d, **kwargs):
        d = d.copy()
        d["elements"] = Element.from_dict(d["elements"])
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
        """Generate a 2D flat matrix array

        Args:
            nx: number of elements in the x direction
            ny: number of elements in the y direction
            pitch: distance between element centers
            kerf: distance between element edges
            units: units of the array dimensions
            impulse_response: impulse response of the elements
            impulse_dt: time step of the impulse response
            id: unique identifier
            name: name of the array
            attrs: additional attributes

        Returns: a Transducer object representing the array
        """
        N = nx * ny
        xpos = (np.arange(nx) - (nx - 1) / 2) * pitch # x positions, centered about x=0
        ypos = (np.arange(ny) - (ny - 1) / 2) * pitch # y positions, centered about y=0
        elements = []
        for i in range(N):
            x = xpos[i % nx] # inner loop through x positions
            y = ypos[i // nx] # outer loop through y positions
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
