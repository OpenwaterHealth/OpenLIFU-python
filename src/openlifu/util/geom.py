import numpy as np
import xarray as xa

from openlifu.util.units import get_ndgrid_from_coords, rescale_coords


def resample_volume(
        volume: xa.DataArray,
        coords: xa.Coordinates,
        matrix: np.ndarray,
        interp_method: str = "linear"
    ) -> xa.DataArray:
    """Resample an xa.DataArray Volume data

    Resample an xa.DataArray Volume to the coordinate system
    specified by coords using the transformation matrix.

    Args:
        volume: xa.DataArray
        coords:
            The coordinate system to which the volume should be resampled
        matrix: np.ndarray
            The transformation matrix in the coords units

    Returns:
        fus.Volume: The transformed Volume object
    """

    prev_units = volume[next(iter(volume.coords.keys()))].units
    coords_units = coords[next(iter(coords.keys()))].units
    volume_rescaled = rescale_coords(volume, coords_units)
    xyz = get_ndgrid_from_coords(coords)
    XYZ = np.append(xyz, np.ones((*xyz.shape[:3], 1)), axis=-1)
    X1 = np.linalg.inv(matrix) @ (matrix @ XYZ)
    interp_method = None
    #TODO: get transform matrix to transform grid
    # inv_matrix = (self.matrix'*self.matrix)\(self.matrix');
    pass
    # XP = np.vstack((np.ravel(Xp).T, np.ones(np.prod(Xp.shape), dtype))
    # XP[:, -1] = 1.0)  # Add a column for homogeneous coordinates

    # inv_matrix = np.linalg.inv(self.matrix.T @ self.matrix)

    # X1 = matrix.dot(XP)
    # X1 = (inv_matrix @ X1).reshape(Xp.shape[:-1], -1, order='F')

    # if options is None:
    #     method = 'linear'
    # else:
    #     method = options['method']

    # data = self.interp((X1.T,), method=method)  # Interpolate the transformed coordinates
    # obj = self.newobj(data, coords, id=self.id,
    #                   name=self.name, matrix=matrix,
    #                   attrs=self.attrs, units=self.units)

    # if len(self):
    #     self.rescale(prev_units)
    rescale_coords(volume, prev_units)

    return volume
