from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Tuple

import numpy as np
import xarray as xa

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin

DEFAULT_ORIGIN = np.zeros(3)


@dataclass
class SolutionAnalysis(DictMixin):
    mainlobe_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Mainlobe PNP", "Peak negative pressure in the mainlobe, in MPa")] = field(default_factory=list)
    """Peak negative pressure in the mainlobe, in MPa"""

    mainlobe_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Mainlobe ISPPA", "Spatial peak pulse average intensity in the mainlobe, in W/cm²")] = field(default_factory=list)
    """Spatial peak pulse average intensity in the mainlobe, in W/cm²"""

    mainlobe_ispta_mWcm2: Annotated[list[float], OpenLIFUFieldData("Mainlobe ISPTA", "Spatial peak time average intensity in the mainlobe, in mW/cm²")] = field(default_factory=list)
    """Spatial peak time average intensity in the mainlobe, in mW/cm²"""

    beamwidth_lat_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB lateral beamwidth", "Lateral beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Lateral beamwidth at -3 dB, in mm"""

    beamwidth_ele_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB elevation beamwidth", "Elevation beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Elevation beamwidth at -3 dB, in mm"""

    beamwidth_ax_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB axial beamwidth", "Axial beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Axial beamwidth at -3 dB, in mm"""

    beamwidth_lat_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB lateral beamwidth", "Lateral beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Lateral beamwidth at -6 dB, in mm"""

    beamwidth_ele_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB elevation beamwidth", "Elevation beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Elevation beamwidth at -6 dB, in mm"""

    beamwidth_ax_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB axial beamwidth", "Axial beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Axial beamwidth at -6 dB, in mm"""

    sidelobe_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Sidelobe PNP", "Peak negative pressure in the sidelobes, in MPa")] = field(default_factory=list)
    """Peak negative pressure in the sidelobes, in MPa"""

    sidelobe_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Sidelobe ISPPA", "Spatial peak pulse average intensity in the sidelobes, in W/cm²")] = field(default_factory=list)
    """Spatial peak pulse average intensity in the sidelobes, in W/cm²"""

    global_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Global PNP", "Maximum peak negative pressure in the entire field, in MPa")] = field(default_factory=list)
    """Maximum peak negative pressure in the entire field, in MPa"""

    global_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Global ISPPA", "Maximum spatial peak pulse average intensity in the entire field, in W/cm²")] = field(default_factory=list)
    """Maximum spatial peak pulse average intensity in the entire field, in W/cm²"""

    p0_Pa: Annotated[list[float], OpenLIFUFieldData("Emitted pressure (Pa)", "Initial pressure values in the field, in Pa")] = field(default_factory=list)
    """Initial pressure values in the field (Pa)"""

    TIC: Annotated[float | None, OpenLIFUFieldData("Thermal index (TIC)", "Thermal index in cranium (TIC)")] = None
    """Thermal index in cranium (TIC)"""

    power_W: Annotated[float | None, OpenLIFUFieldData("Emitted Power (W)", "Emitted power from the transducer face (W)")] = None
    """Emitted power from the transducer face (W)"""

    MI: Annotated[float | None, OpenLIFUFieldData("Mechanical index (MI)", "Mechanical index (MI)")] = None
    """Mechanical index (MI)"""

    global_ispta_mWcm2: Annotated[float | None, OpenLIFUFieldData("Global ISPTA (mW/cm²)", "Global Intensity at Spatial-Peak, Time-Average (I_SPTA) (mW/cm²)")] = None
    """Global Intensity at Spatial-Peak, Time-Average (I_SPTA) (mW/cm²)"""

    @staticmethod
    def from_json(json_string : str) -> SolutionAnalysis:
        """Load a SolutionAnalysis from a json string"""
        return SolutionAnalysis.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a SolutionAnalysis to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete SolutionAnalysis object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)


@dataclass
class SolutionAnalysisOptions(DictMixin):
    standoff_sound_speed: Annotated[float, OpenLIFUFieldData("Standoff sound speed (m/s)", "Speed of sound in standoff, for calculating initial impedance")] = 1500.0
    """Speed of sound in standoff, for calculating initial impedance"""

    standoff_density: Annotated[float, OpenLIFUFieldData("Standoff density (kg/m³)", "Density of standoff medium (kg/m³)")] = 1000.0
    """Density of standoff medium (kg/m³)"""

    ref_sound_speed: Annotated[float, OpenLIFUFieldData("Reference sound speed (m/s)", "Reference speed of sound in the medium (m/s)")] = 1500.0
    """Reference speed of sound in the medium (m/s)"""

    ref_density: Annotated[float, OpenLIFUFieldData("Reference density (kg/m³)", "Reference density (kg/m³)")] = 1000.0
    """Reference density (kg/m³)"""

    mainlobe_aspect_ratio: Annotated[Tuple[float, float, float], OpenLIFUFieldData("Mainlobe aspect ratio (lat,ele,ax)", "Aspect ratio of the mainlobe mask")] = (1., 1., 5.)
    """Aspect ratio of the mainlobe ellipsoid mask, in the form (lat,ele,ax). (1,1,5) means an ellipsoid 5x as long as it is wide."""

    mainlobe_radius: Annotated[float, OpenLIFUFieldData("Mainlobe mask radius", "Size of the mainlobe mask, in the units provided for Distance units (`distance_units`)")] = 2.5e-3
    """Size of the mainlobe mask, in the units provided for Distance units (`distance_units`). The mainlobe mask is an ellipsoid with this radius, scaled by the `mainlobe_aspect_ratio`."""

    beamwidth_radius: Annotated[float, OpenLIFUFieldData("Beamwidth search radius", "Size of the beamwidth search, in the units provided for Distance units (`distance_units`)")] = 5e-3
    """Size of the beamwidth search, in the units provided for Distance units (`distance_units`). The beamwidth is found along the lateral and elevation lines perpendicular to the focus axis."""

    sidelobe_radius: Annotated[float, OpenLIFUFieldData("Sidelobe radius", "Size of the sidelobe mask, in the units provided for Distance units (`distance_units`)")] = 3e-3
    """Size of the sidelobe mask, in the units provided for Distance units (`distance_units`). Pressure outside of this ellipsoid (scaled by `mainlobe_aspect_ratio`) is considered outside of the focal region."""

    sidelobe_zmin: Annotated[float, OpenLIFUFieldData("Sidelobe minimum z", "Minimum z coordinate of the sidelobe mask, in the units provided for Distance units (`distance_units`)")] = 1e-3
    """Minimum z coordinate of the sidelobe mask, in the units provided for Distance units (`distance_units`). This value is used to ignore emitted pressure artifacts."""

    distance_units: Annotated[str, OpenLIFUFieldData("Distance units", "The units used for distance measurements")] = "m"
    """The units used for distance measurements"""

def find_centroid(da: xa.DataArray, cutoff:float) -> np.ndarray:
    """Find the centroid of a thresholded region of a DataArray"""
    da = da.where(da > cutoff, 0)
    dims = list(da.dims)
    coords = np.meshgrid(*[da.coords[coord] for coord in dims], indexing='ij')
    centroid = np.array([np.sum(da*coords[dims.index(dim)])/da.sum() for dim in da.dims])
    return centroid

def get_focus_matrix(focus, origin=[0,0,0]) -> np.ndarray:
    """Get the coordinate transform from the focus to the transducer

    The "focal coordinate system" here refers to a coordinate system whose z-axis is along
    the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
    to the focus center.

    Args:
        focus: A 3D point describing the focal coordinate system location in transducer coordinates
        origin: A 3D point describing the transducer "effective origin" in transducer coordinates

    Returns: A 4x4 affine transform matrix that describes the coordinate transformation from the focus
        coordinate system to the transducer coordinates.
    """
    focus = np.array(focus)
    origin = np.array(origin)
    zvec = (focus-origin)/np.linalg.norm(focus-origin)
    az = -np.arctan2(zvec[0],zvec[2])
    xvec = np.array([np.cos(az), 0, np.sin(az)])
    yvec = np.cross(zvec, xvec)
    uv = np.array([xvec, yvec, zvec, focus])
    M = np.concatenate([uv.T,np.zeros([1,4])],axis=0)
    M[3,3] = 1
    return M

def get_gridded_transformed_coords(da: xa.DataArray, matrix: np.ndarray, as_dataset=True):
    """Transform the coords of a DataArray using a transform matrix.

    Args:
        da: DataArray whose coordinates will be used
        matrix: a 4x4 coordinate transformation matrix, transforming from the desired coordinate system
            to the coordinate system of `da`
        as_dataset: Whether to return the transformed coords as a numpy array or an xarray Dataset

    Returns the transformed coordinate grid as either a numpy array or an xarray Dataset.
    """
    shape = tuple(da.sizes[d] for d in da.dims)
    coords = np.meshgrid(*[da.coords[coord] for coord in da.dims], indexing='ij')
    coords = np.concatenate([coord.reshape(-1,1) for coord in coords], axis=1)
    coords = np.concatenate([coords, np.ones((coords.shape[0],1))], axis=1)
    coords = np.dot(np.linalg.inv(matrix), coords.T).T
    coords = coords[:,0:3].reshape(*shape, 3)
    if as_dataset:
        coords = xa.Dataset({f'd_{dim}': (da.dims, coords[...,i]) for i, dim in enumerate(da.dims)}, coords=da.coords)
    return coords

def get_offset_grid(da: xa.DataArray, focus, origin=DEFAULT_ORIGIN, as_dataset=True):
    """Transform the coords of a DataArray that is in transducer coordinates to focus coordinates

    See `get_focus_matrix` for the meaning of "focus coordinates"

    Args:
        da: DataArray whose coordinates will be used (presumably the transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        as_dataset: Whether to return the transformed coords as a numpy array or an xarray Dataset

    Returns the transformed coordinate grid as either a numpy array or an xarray Dataset.
    """
    M = get_focus_matrix(focus, origin=origin)
    #M = focus.get_matrix()
    coords = get_gridded_transformed_coords(da, M, as_dataset=as_dataset)
    return coords

def calc_dist_from_focus(da: xa.DataArray, focus, origin=DEFAULT_ORIGIN, aspect_ratio=[1,1,1], as_dataarray=True):
    """Compute a distance map from a focus point in transducer space, using a possibly distorted metric that respects
    the symmetry of the focus shape (e.g. it could be cigar-shaped).

    Args:
        da: DataArray that will supply the coordnate grid (presumably transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        aspect_ratio: x,y,z scalings on the focal coordinate system to distort the space before computing distance
            (see `get_focus_matrix` for the meaning of "focus coordinates").
        as_dataarray: Whether to return the distance map as a numpy array or an xarray DataArray

    Returns the distance map as either a numpy array or an xarray DataArray.
    """
    coords = get_offset_grid(da, focus, origin=origin, as_dataset=False)
    dist = np.sqrt(np.sum((coords/aspect_ratio)**2, axis=-1))
    if as_dataarray:
        dist = xa.DataArray(dist, coords=da.coords, dims=da.dims)
    return dist

def get_mask(
    da: xa.DataArray,
    focus,
    distance:float,
    origin=DEFAULT_ORIGIN,
    aspect_ratio=[1,1,1],
    operator='<',
) -> xa.DataArray:
    """Compute a boolean mask of the focus region in transducer space.

    The focus region is an ellipsoid centered at the focus point.

    Args:
        da: DataArray that will supply the coordnate grid (presumably transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        distance: How far from the `focus` to include points in the mask. See `calc_dist_from_focus`
            for the distorted metric under which a ball of points becomes an ellispoid in euclidean space.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        aspect_ratio: x,y,z scalings on the focal coordinate system to distort the space before computing distances
            (see `get_focus_matrix` for the meaning of "focus coordinates").
        operator: a string representation of an inequality operator that represents the desired masking operation.
            The default '<' for example includes points strictly *inside* the focal ellipsoid.

    Returns: the focal mask as an xarray DataArray
    """
    dist = calc_dist_from_focus(da, focus, origin=origin, aspect_ratio=aspect_ratio)
    if operator == '<':
        mask = dist < distance
    elif operator == '<=':
        mask = dist <= distance
    elif operator == '>':
        mask = dist > distance
    elif operator == '>=':
        mask = dist >= distance
    else:
        raise ValueError("Operator must be '<', '>', '<=', or '>='.")
    return mask

def interp_transformed_axis(
    da: xa.DataArray,
    focus,
    dim,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None,
) -> xa.DataArray:
    """Interpolate data along a focal coordinate system axis.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be sampled (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns: a 1D DataArray of interpolated values from `da`.
    """
    matrix = get_focus_matrix(focus, origin=origin)
    coords = get_gridded_transformed_coords(da, matrix, as_dataset=True)
    if min_offset is None:
        min_offset = float(coords[f'd_{dim}'].min())
    if max_offset is None:
        max_offset = float(coords[f'd_{dim}'].max())
    n = da.sizes[dim]*2
    interp_dim = np.linspace(min_offset, max_offset, n)
    interp_xyz = np.zeros((n,3))
    interp_xyz[:,da.dims.index(dim)] = interp_dim
    interp_xyz = np.dot(matrix, np.concatenate([interp_xyz.T, np.ones((1,n))], axis=0)).T[:,0:3]
    interpolants = {d: xa.DataArray(interp_xyz[:,i], coords={f'offset_d{dim}':interp_dim}) for i, d in enumerate(da.dims)}
    interp_da = da.interp(**interpolants)
    for d in da.dims:
        interp_da.assign_coords(dim=(f'offset_d{dim}', interpolants[d].to_numpy()))
    return interp_da

def get_beam_bounds(
    da: xa.DataArray,
    focus,
    dim,
    cutoff:float,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None,
) -> Tuple[float, float]:
    """Determine how far along a focal coordinate system axis a DataArray's value stays above a certain cutoff.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be considered (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        cutoff: The threshold against which `da` values are compared.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns:
        negoff: Distance from the `focus` along the negative focal `dim` direction until `da` first goes below `cutoff`.
        posoff: Distance from the `focus` along the positive focal `dim` direction until `da` first goes below `cutoff`.
    """
    interp_da = interp_transformed_axis(da, focus=focus, dim=dim, origin=origin, min_offset=min_offset, max_offset=max_offset)
    offset = interp_da.coords[f'offset_d{dim}']
    da_negoff = interp_da.where(offset <= 0, drop=True)
    da_posoff = interp_da.where(offset >= 0, drop=True)
    da_negoff = da_negoff.where(da_negoff < float(cutoff), drop=True)
    if da_negoff.size > 0:
        negoff = float(da_negoff.coords[f'offset_d{dim}'][-1])
    else:
        negoff = np.nan
    da_posoff = da_posoff.where(da_posoff < float(cutoff), drop=True)
    if da_posoff.size > 0:
        posoff = float(da_posoff.coords[f'offset_d{dim}'][0])
    else:
        posoff = np.nan
    return negoff, posoff

def get_beamwidth(
    da: xa.DataArray,
    focus,
    dim,
    cutoff:float | None=None,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None
) -> float:
    """Determine the FWHM (or differently thresholded width) of a DataArray along a focal coordinate system axis.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be considered (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        cutoff: The threshold against which `da` values are compared. If not provided then the half-max is used, making
            this an "FWHM" function.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns: The "beam width" along the focal `dim` direction, by measuring from the from closest-to-focus point along the
        *negative* focal `dim` axis for which `da` goes below `cutoff` up to the closest-to-focus point along the
        *positive* focal `dim` axis for which `da` goes below `cutoff`.
    """
    if cutoff is None:
        cutoff = float(da.max())/2
    negoff, posoff = get_beam_bounds(da, focus=focus, dim=dim, cutoff=float(cutoff), origin=origin, min_offset=min_offset, max_offset=max_offset)
    bw = posoff - negoff
    return bw
