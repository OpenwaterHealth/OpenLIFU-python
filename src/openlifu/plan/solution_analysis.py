import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import xarray as xa

from openlifu.util.dict_conversion import DictMixin

DEFAULT_ORIGIN = np.zeros(3)


@dataclass
class SolutionAnalysis(DictMixin):
    mainlobe_pnp_MPa: list[float] = field(default_factory=list)
    mainlobe_isppa_Wcm2: list[float] = field(default_factory=list)
    mainlobe_ispta_mWcm2: list[float] = field(default_factory=list)
    beamwidth_lat_3dB_mm: list[float] = field(default_factory=list)
    beamwidth_ele_3dB_mm: list[float] = field(default_factory=list)
    beamwidth_ax_3dB_mm: list[float] = field(default_factory=list)
    beamwidth_lat_6dB_mm: list[float] = field(default_factory=list)
    beamwidth_ele_6dB_mm: list[float] = field(default_factory=list)
    beamwidth_ax_6dB_mm: list[float] = field(default_factory=list)
    sidelobe_pnp_MPa: list[float] = field(default_factory=list)
    sidelobe_isppa_Wcm2: list[float] = field(default_factory=list)
    global_pnp_MPa: list[float] = field(default_factory=list)
    global_isppa_Wcm2: list[float] = field(default_factory=list)
    p0_Pa: list[float] = field(default_factory=list)
    TIC: Optional[float] = None
    power_W: Optional[float] = None
    MI: Optional[float] = None
    global_ispta_mWcm2: Optional[float] = None

    @staticmethod
    def from_json(json_string : str) -> "SolutionAnalysis":
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
    standoff_sound_speed: float = 1500.0
    standoff_density: float = 1000.0
    ref_sound_speed: float = 1500.0
    ref_density: float = 1000.0
    focus_diameter: float = 0.5
    mainlobe_aspect_ratio: Tuple[float, float, float] = (1., 1., 5.)
    mainlobe_radius: float = 2.5e-3
    beamwidth_radius: float = 5e-3
    sidelobe_radius: float = 3e-3
    sidelobe_zmin: float = 1e-3
    distance_units: str = "m"

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
    min_offset:Optional[float]=None,
    max_offset:Optional[float]=None,
) -> xa.DataArray:
    """Interpolate data along the focal axis.

    Here the *focal axis* is the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
    to the focus center.

    Args:
        da: DataArray that will supply the coordnate grid (presumably transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            See `get_focus_matrix` for the meaning of "focal coordinate system." For example, the "axial" dimension of
            a transducer corresponds to the focal axis in the focal coordinate system.
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
    min_offset:Optional[float]=None,
    max_offset:Optional[float]=None,
) -> Tuple[float, float]:
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

def get_beamwidth(da: xa.DataArray, focus, dim, cutoff=None, origin=DEFAULT_ORIGIN, min_offset=None, max_offset=None):
    if cutoff is None:
        cutoff = float(da.max())/2
    negoff, posoff = get_beam_bounds(da, focus=focus, dim=dim, cutoff=float(cutoff), origin=origin, min_offset=min_offset, max_offset=max_offset)
    bw = posoff - negoff
    return bw
