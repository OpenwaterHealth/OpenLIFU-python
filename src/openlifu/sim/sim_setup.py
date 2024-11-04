import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import xarray as xa

from openlifu.geo import Point
from openlifu.io.dict_conversion import DictMixin
from openlifu.seg import SegmentationMethod
from openlifu.util.units import getunitconversion, rescale_coords
from openlifu.xdc import Transducer


@dataclass
class SimSetup(DictMixin):
    dims: Tuple[str, str, str] = ("lat", "ele", "ax")
    names: Tuple[str, str, str] = ("Lateral", "Elevation", "Axial")
    spacing: float = 1.0
    units: str = "mm"
    x_extent: Tuple[float, float] = (-30., 30.)
    y_extent: Tuple[float, float] = (-30., 30.)
    z_extent: Tuple[float, float] = (-4., 60.)
    dt: float = 0.
    t_end: float = 0.
    c0: float = 1500.0
    cfl: float = 0.5
    options: dict = field(default_factory=dict)

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

    def get_coords(self, dims=None, units: Optional[str] = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        sizes = self.get_size(dims)
        extents = self.get_extent(dims, units)
        coords = xa.Coordinates({dim: np.linspace(extents[i][0], extents[i][1], sizes[i]) for i, dim in enumerate(dims)})
        for i, dim in enumerate(dims):
            coords[dim].attrs['units'] = units
            coords[dim].attrs['long_name'] = self.names[i]
        return coords

    def get_corners(self, id: str = "corners", units: Optional[str] = None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        xyz = np.array(np.meshgrid(self.x_extent, self.y_extent, self.z_extent, indexing='ij'))
        corners = xyz.reshape(3,-1)
        return corners*scl

    def get_extent(self, dims: Optional[str]=None, units: Optional[str] = None):
        dims = self.dims if dims is None else dims
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        extents = [self.x_extent, self.y_extent, self.z_extent]
        return np.array([extents[self.dims.index(dim)] for dim in dims])*scl

    def get_max_cycle_offset(self, arr:Transducer, frequency: Optional[float] = None, delays: Optional[np.ndarray]=None, zmin: float =10e-3):
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

    def get_max_distance(self, arr: Transducer, units: Optional[str] = None):
        units = self.units if units is None else units
        corners = self.get_corners(units=units)
        distances = np.array([[el.distance_to_point(corner, units=units) for corner in corners.T] for el in arr.elements])
        max_distance = np.max(distances)
        return max_distance

    def get_size(self, dims: Optional[str]=None):
        dims = self.dims if dims is None else dims
        n = [int(np.round(np.diff(ext)/self.spacing))+1 for ext in [self.x_extent, self.y_extent, self.z_extent]]
        return np.array([n[self.dims.index(dim)] for dim in dims]).squeeze()

    def get_spacing(self, units: Optional[str] = None):
        units = self.units if units is None else units
        return getunitconversion(self.units, units)*self.spacing

    #TODO: since we don't have a concept of scene here, and that the simulation scene is needed in protocol.calc_solution,
    # we will return each prepared object one-by-one.
    #TODO: Missing the "markers" from matlab scene in "scene.transform(self, coords, matrix, options)"
    #TODO: The arg transform needs to be added since transducer.matrix does not exists anymore
    def setup_sim_scene(
            self,
            transducer: Transducer,
            transform: np.ndarray,
            target: Point,
            seg_method: SegmentationMethod,
            volume: xa.DataArray = None,
            interp_method: str = "Linear",
            units: Optional[str] = None
        ) -> Tuple[xa.DataArray, Transducer, Point]:
        """ Prepare a simulation scene composed of a volume, transducer and targets.

        Setup a simulation scene with a volume, transducer and target point.
        All objects are resampled to the geo-referenced simulation grid (lon, lat, ele).
        For the volume, a segmentation is also performed that defines the simulation medium.

        Args:
            volume: xa.DataArray
            transducer: xdc.Transducer
            transform: np.ndarray
            target: geo.Point
            seg_method: seg.SegmentationMethod
            interp_method: str
                Interpolation method for the simulation grid (Default: \"Linear\").
            units: str
                Units of simulation grid (Default: self.units).

        Returns
            params: The resampled and segmented xa.DataArray volume to the simulation grid
            transducer_tr: The resampled xdc.Transducer to the simulation grid
            target_tr: The resampled geo.Point to the simulation grid
        """
        units = self.units if units is None else units
        sim_coords = self.get_coords(units=units)
        #TODO: in a near future, the following will transform transducer and target. Currently Slicer does it.
        # sim_matrix = convert_transform(transform, units=transducer.units, tgt_units=units)  #TODO: not clear here where the transform should come from since transducer.get_matrix() does not exist
        # rescaled_affine = convert_transform(volume.affine, units=volume.affine_units, tgt_units=units)
        # volume = volume.assign_attrs(affine_units=units, affine=rescaled_affine)
        # volume_resampled = resample_volume(volume, sim_coords, sim_matrix, interp_method)
        # transducer_tr = transducer.transform(sim_matrix, units)
        # target_tr = target.transform(sim_matrix, units=units, new_dims=sim_coords.dims)
        transducer.rescale(units)
        target.rescale(units)
        if volume is None:
            params = seg_method.ref_params(sim_coords)
        else:
            sim_coords_units = sim_coords[next(iter(sim_coords.keys()))].units
            volume_resampled = rescale_coords(volume, sim_coords_units)
            params = seg_method.seg_params(volume_resampled)  #TODO: currently only "water" method is tested with db-simple-example-v04
                                                              # we should have a whole MRI segmentation instead.

        return params, transducer, target
