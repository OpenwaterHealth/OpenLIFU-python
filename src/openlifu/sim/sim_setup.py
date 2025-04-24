from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Tuple

import numpy as np
import xarray as xa

from openlifu.geo import Point
from openlifu.seg import SegmentationMethod
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer


@dataclass
class SimSetup(DictMixin):
    dims: Annotated[Tuple[str, str, str], OpenLIFUFieldData("Dimension keys", "Codenames of the axes in the coordinate system being used")] = ("lat", "ele", "ax")
    """Names of the axes in the coordinate system being used"""

    names: Annotated[Tuple[str, str, str], OpenLIFUFieldData("Dimension names", "Human readable names of the axes in the coordinate system being used")] = ("Lateral", "Elevation", "Axial")
    """"Human readable names of the axes in the coordinate system being used"""

    spacing: Annotated[float, OpenLIFUFieldData("Spacing", "Simulation grid spacing")] = 1.0
    """Simulation grid spacing"""

    units: Annotated[str, OpenLIFUFieldData("Spatial units", "Units used for spatial measurements")] = "mm"
    """Units used for spatial measurements"""

    x_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("X-extent", "Simulation grid extent along the first dimension")] = (-30., 30.)
    """Simulation grid extent along the first dimension"""

    y_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("Y-extent", "Simulation grid extend along the second dimension")] = (-30., 30.)
    """Simulation grid extend along the second dimension"""

    z_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("Z-extent", "Simulation grid extend along the third dimension")] = (-4., 60.)
    """Simulation grid extend along the third dimension"""

    dt: Annotated[float, OpenLIFUFieldData("Time step", "Simulation time step")] = 0.
    """Simulation time step"""

    t_end: Annotated[float, OpenLIFUFieldData("End time", """Simulation end time""")] = 0.
    """Simulation end time"""

    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Reference speed of sound for converting distance to time")] = 1500.0
    """Reference speed of sound for converting distance to time"""

    cfl: Annotated[float, OpenLIFUFieldData("CFL number", "Courant-Friedrichs-Lewy number")] = 0.5
    """Courant-Friedrichs-Lewy number"""

    options: Annotated[dict[str, str], OpenLIFUFieldData("Simulation options", "Additional simulation options")] = field(default_factory=dict)
    """Additional simulation options"""

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
        nx = np.diff(self.x_extent)/self.spacing
        x_extent = tuple(np.arange(2)*np.round(nx)*self.spacing + self.x_extent[0])
        if ((0.5-np.abs((nx % 1) - 0.5))/ np.round(nx)) > 1e-3:
            logging.warning(f"x_extent {self.x_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {x_extent}.")
        self.x_extent = x_extent
        ny = np.diff(self.y_extent)/self.spacing
        y_extent = tuple(np.arange(2)*np.round(ny)*self.spacing + self.y_extent[0])
        if ((0.5-np.abs((ny % 1) - 0.5))/ np.round(ny)) > 1e-3:
            logging.warning(f"y_extent {self.y_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {y_extent}.")
        self.y_extent = y_extent
        nz = np.diff(self.z_extent)/self.spacing
        z_extent = tuple(np.arange(2)*np.round(nz)*self.spacing + self.z_extent[0])
        if ((0.5-np.abs((nz % 1) - 0.5))/ np.round(nz)) > 1e-3:
            logging.warning(f"z_extent {self.z_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {z_extent}.")
        self.z_extent = z_extent

    def get_coords(self, dims=None, units: str | None = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        sizes = self.get_size(dims)
        extents = self.get_extent(dims, units)
        coords = xa.Coordinates({dim: np.linspace(extents[i][0], extents[i][1], sizes[i]) for i, dim in enumerate(dims)})
        for i, dim in enumerate(dims):
            coords[dim].attrs['units'] = units
            coords[dim].attrs['long_name'] = self.names[i]
        return coords

    def get_corners(self, id: str = "corners", units: str | None = None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        xyz = np.array(np.meshgrid(self.x_extent, self.y_extent, self.z_extent, indexing='ij'))
        corners = xyz.reshape(3,-1)
        return corners*scl

    def get_extent(self, dims: str | None=None, units: str | None = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        extents = [self.x_extent, self.y_extent, self.z_extent]
        return np.array([extents[self.dims.index(dim)] for dim in dims])*scl

    def get_max_cycle_offset(self, arr:Transducer, frequency: float | None = None, delays: np.ndarray | None=None, zmin: float =10e-3):
        frequency = arr.frequency if frequency is None else frequency
        delays = np.zeros(arr.numelements()) if delays is None else delays
        coords = self.get_coords(units="m")
        cvals = [coords['lat'], coords['ele'], coords['ax'].sel(ax=slice(zmin, None))]
        ndg = np.meshgrid(*cvals)
        dists = [np.sqrt((ndg[0]-pos[0])**2 + (ndg[1]-pos[1])**2 + (ndg[2]-pos[2])**2) for pos in arr.get_positions(units="m")]
        tof = [dist/self.c0 + delays[i] for i, dist in enumerate(dists)]
        dtof = np.array(tof).max(axis=0) - np.array(tof).min(axis=0)
        max_cycle_offset = dtof.max()*frequency
        return max_cycle_offset

    def get_max_distance(self, arr: Transducer, units: str | None = None):
        units = self.units if units is None else units
        corners = self.get_corners(units=units)
        distances = np.array([[el.distance_to_point(corner, units=units) for corner in corners.T] for el in arr.elements])
        max_distance = np.max(distances)
        return max_distance

    def get_size(self, dims: str | None=None):
        dims = self.dims if dims is None else dims
        n = [int((np.round(np.diff(ext)/self.spacing)).item())+1 for ext in [self.x_extent, self.y_extent, self.z_extent]]
        return np.array([n[self.dims.index(dim)] for dim in dims]).squeeze()

    def get_spacing(self, units: str | None = None):
        units = self.units if units is None else units
        return getunitconversion(self.units, units)*self.spacing

    def setup_sim_scene(
            self,
            seg_method: SegmentationMethod,
            volume: xa.DataArray | None = None
        ) -> Tuple[xa.DataArray, Transducer, Point]:
        """ Prepare a simulation scene composed of a simulation grid

        Setup a simulation scene with a simulation grid including physical properties.
        A segmentation is performed to detect the medium, so we can assign
        physical properties to each voxel, later used by the ultrasound simulation.
        This assume that the input volume is resampled to the geo-referenced simulation grid (lon, lat, ele).

        Args:
            seg_method: seg.SegmentationMethod
            volume: xa.DataArray
                Optional volume to be used for simulation grid definition (Default: None).
                The volume is assumed to be resampled on sim grid coordinates.

        Returns
            params: The xa.DataArray simulation grid with physical properties for each voxel
        """
        if volume is None:
            sim_coords = self.get_coords()
            params = seg_method.ref_params(sim_coords)
        else:
            params = seg_method.seg_params(volume)

        return params
