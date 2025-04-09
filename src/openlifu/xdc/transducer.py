from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

import numpy as np
import vtk

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion

from .element import Element


@dataclass
class Transducer:
    id: Annotated[str, OpenLIFUFieldData("Transducer ID", "Unique identifier for transducer")] = "transducer"
    """Unique identifier for transducer"""

    name: Annotated[str, OpenLIFUFieldData("Transducer name", "Human readable name for transducer")] = ""
    """Human readable name for transducer"""

    elements: Annotated[List[Element], OpenLIFUFieldData("Elements", "Collection of transducer Elements")] = field(default_factory=list)
    """Collection of transducer Elements"""

    frequency: Annotated[float, OpenLIFUFieldData("Frequency (Hz)", "Nominal array frequency (Hz)")] = 400.6e3
    """Nominal array frequency (Hz)"""

    units: Annotated[str, OpenLIFUFieldData("Units", "Native units of transducer local coordinate space")] = "m"
    """Native units of transducer local coordinate space"""

    attrs: Annotated[Dict[str, Any], OpenLIFUFieldData("Attributes", "Additional transducer attributes")] = field(default_factory=dict)
    """Additional transducer attributes"""

    registration_surface_filename: Annotated[str | None, OpenLIFUFieldData("Registration surface filename", "Relative path to an open surface of the transducer to be used for registration")] = None
    """Relative path to an open surface of the transducer to be used for registration"""

    transducer_body_filename: Annotated[str | None, OpenLIFUFieldData("Transducer body filename", "Relative path to the closed surface mesh for visualizing the transducer body")] = None
    """Relative path to the closed surface mesh for visualizing the transducer body"""

    standoff_transform: Annotated[np.ndarray, OpenLIFUFieldData("Standoff transform", "Affine transform representing the way in which the standoff for this transducer displaces the transducer.\n\nA \"standoff transform\" applies a displacement in transducer space that moves a transducer to where it would\nbe situated with the standoff in place. The idea is that if you start with a transform that places a transducer\ndirectly against skin, then pre-composing that transform by a \"standoff transform\" serves to nudge the transducer\nsuch that there is space for the standoff to be between it and the skin.\n\nSee also `openlifu.geo.create_standoff_transform`.\n\nThe units of this transform are assumed to be the native units of the transducer, the `Transducer.units` field.")] = field(default_factory=lambda: np.eye(4, dtype=float))
    """Affine transform representing the way in which the standoff for this transducer displaces the transducer.

    A "standoff transform" applies a displacement in transducer space that moves a transducer to where it would
    be situated with the standoff in place. The idea is that if you start with a transform that places a transducer
    directly against skin, then pre-composing that transform by a "standoff transform" serves to nudge the transducer
    such that there is space for the standoff to be between it and the skin.

    See also `openlifu.geo.create_standoff_transform`.

    The units of this transform are assumed to be the native units of the transducer, the `Transducer.units` field.
    """

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
             transform:np.ndarray | None=None,
             units:str | None=None,
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

    def get_actor(self, transform:np.ndarray | None=None, units:str | None=None, facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        polydata = self.get_polydata(units=units, transform=transform, facecolor=facecolor)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def get_polydata(self, transform:np.ndarray | None=None, units:str | None=None, facecolor=None):
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

    def get_corners(self, transform:np.ndarray | None=None, units:str | None=None):
        units = self.units if units is None else units
        matrix = transform if transform is not None else np.eye(4)
        return [element.get_corners(units=units, matrix=matrix) for element in self.elements]

    def get_effective_origin(self, apodizations:np.ndarray, units:str | None=None):
        """Get the centroid of the effective active region of the transducer based on apodizations.

        Args:
            apodizations: vector of apodizations for the transducer elements
            units: units in which to describe the centroid. If not provided then transducer native units are used.

        Returns: a 3-element array describing the centroid in the transducer coordinate system
        """
        units = self.units if units is None else units
        return (apodizations.reshape(-1,1) * self.get_positions(units=units)).sum(axis=0)/apodizations.sum()

    def get_positions(self, transform:np.ndarray | None=None, units:str | None=None):
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

    def get_standoff_transform_in_units(self, units:str) -> np.ndarray:
        """Get the transducer's standoff transform in the desired units."""
        matrix = self.standoff_transform.copy()
        matrix[0:3, 3] *= getunitconversion(self.units, units)
        return matrix

    @staticmethod
    def merge(list_of_transducers:List[Transducer]) -> Transducer:
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
        d["standoff_transform"] =  d["standoff_transform"].tolist()
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
        d["elements"] = [Element.from_dict(element) for element in d["elements"]]
        if "standoff_transform" in d:
            d["standoff_transform"] = np.array(d["standoff_transform"])
        return Transducer(**d, **kwargs)

    @staticmethod
    def from_json(json_string : str) -> Transducer:
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
    def gen_matrix_array(nx:int=2,
                         ny:int=2,
                         pitch:float=1.0,
                         kerf:float=0.0,
                         units:str="mm",
                         impulse_response:float|np.ndarray=1.0,
                         impulse_dt:float=1.0,
                         id:str='array',
                         name:str='Array',
                         attrs:Dict|None=None):
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
        if attrs is None:
            attrs = {}
        return Transducer(elements=elements, id=id, name=name, attrs=attrs)
