from typing import Dict, Any
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from xarray import DataArray

from openlifu.util.units import getunitconversion
from openlifu.geo import Point
from openlifu.bf import offset_grid


@dataclass
class BeamwidthResult:
    dims: tuple
    beamwidth: float
    units: str
    inlier_mask: np.ndarray = None
    fit_mask: np.ndarray = None
    inlier_points: np.ndarray = None
    fit_points: np.ndarray = None
    inlier_hull: np.ndarray = None
    fit_hull: np.ndarray = None


def get_beamwidth(vol: DataArray, coords_units: str, focus: Point, cutoff: float, options: Dict[str, Any]) -> BeamwidthResult:
    """
    Calculates the beam width of a volume at a given point.

    Args:
        vol: xarray.DataArray
            The input DataArray.
        coords_units: str
            The unit of data coordinates.
        focus : Point
            The focus point to be used as reference.
        cutoff: float
            Cutoff value for the volume data.
        options: dictionary
            Additional parameters:
                - 'units': str
                    Target units. Default focus.units.
                - 'dims': tuple
                    Dimension ids to operate on. Defaults (0, 1).
                - 'mask': np.ndarray[bool]
                    Search mask.
                - 'hulls': bool
                    Whether to use convex hulls. Defaults False.
                - 'points': bool
                    Whether to output a list of points for the beam width. Defaults False.
                - 'masks': np.ndarray[bool]
                    Whether to output a mask for the beam width. Defaults False.
                - 'simplify_hulls': bool
                    Whether to simplify the convex hull using Delaunay. Defaults True.

    Returns:
        A dictionary containing beam width, units and optionally other data.
    """
    # Define default values for options if not provided
    if not hasattr(options, 'units') or options['units'] is None:
        options['units'] = focus.units
    if not hasattr(options, 'dims') or options['dims'] is None:
        options['dims'] = (0, 1)
    if not hasattr(options, 'hulls') or options['hulls'] is None:
        options['hulls'] = False
    if not hasattr(options, 'points') or options['points'] is None:
        options['points'] = False
    if not hasattr(options, 'masks') or options['masks'] is None:
        options['masks'] = False
    if not hasattr(options, 'simplify_hulls') or options['simplify_hulls'] is None:
        options['simplify_hulls'] = True

    # Get coordinates of volume
    coords = vol.coords

    scale = getunitconversion(coords_units, options['units'])
    coords_rescaled = [coord * scale for coord in coords]  # equivalent of coords.rescale(args["units"])

    if options['mask'] is None:
        search_mask = np.ones(coords_rescaled.shape)
    else:
        search_mask = options['mask']

    ngrid0 = np.meshgrid(*coords, indexing='ij')
    mdata = search_mask * vol.data
    inlier_mask = mdata > cutoff
    ogrid = offset_grid(coords, coords_units, focus, units="mm")
    omask = [ogrid[..., ii][inlier_mask] for ii in range(ogrid.shape[-1])]  #TODO: better to use vectorization here omask = ogrid[inlier_mask]
    inlier_points = [ngrid0[ii][inlier_mask] for ii in range(len(ngrid0))]
    inlier_points = np.stack(inlier_points, axis=-1)

    try:
        # Create convex hull(s) from the set of inlier points
        if options['simplify_hulls']:
            inlier_hull = ConvexHull(inlier_points)  #TODO: simplification not implemented in scipy ConvexHull
        else:
            inlier_hull = ConvexHull(inlier_points)
    except QhullError:
        # If convex hull creation fails (e.g., too few points), add jitter and try again
        print("Invalid inliers, attempting to add jitter to create a valid volume...")  #TODO: should be using self.logger
        minmax_coords = np.array([(np.min(coords[i]), np.max(coords[i])) for i in range(len(coords))])  #TODO: min-max should be from coords.extent
        coords_shape = tuple([len(coords[i]) for i in range(len(coords))])
        dx = np.mean(np.diff(minmax_coords) / (np.array(coords_shape) - 1))
        inlier_points = inlier_points + (np.random.rand(*inlier_points.shape) - 0.5)*dx/2
        # Create convex hull(s) from the set of inlier points
        if options['simplify_hulls']:
            inlier_hull = ConvexHull(inlier_points)  #TODO: simplification not implemented in scipy ConvexHull
            # inlier_hull = Delaunay(inlier_points)
        else:
            inlier_hull = ConvexHull(inlier_points)

    hull_indices = np.unique(inlier_hull.simplices)
    hull_points = np.stack([omask[i][inlier_hull.vertices] for i in range(len(omask))], axis=-1)
    omat = [hull_points[:, i:(i+1)] - hull_points[:, i:(i+1)].T for i in range(hull_points.shape[-1])]
    omat = np.stack([omat[i] for i in options['dims']], axis=-1)
    dists = np.sqrt(np.sum(omat**2, axis=-1))
    beamwidth = np.max(dists)
    d_dims = np.sqrt(np.sum(ogrid[..., options['dims']]**2, axis=-1))
    fit_mask = d_dims <= (beamwidth/2)

    res = BeamwidthResult(
        dims=options['dims'],
        beamwidth=beamwidth,
        units=options['units']
    )
    if options['masks']:
        res.inlier_mask = inlier_mask
        res.fit_mask = fit_mask
    if options['points'] or options['hulls']:
        res.inlier_points = inlier_points
        res.fit_points = [ngrid0[ii][fit_mask] for ii in range(len(ngrid0))]
    if options['hulls']:
        res.inlier_hull = inlier_hull
        res.fit_hull = ConvexHull(np.stack(res.fit_points, axis=-1))  #TODO: simplification not implemented in scipy ConvexHull

    return res