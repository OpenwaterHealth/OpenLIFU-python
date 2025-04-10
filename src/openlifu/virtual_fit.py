from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Sequence, Tuple

import numpy as np

from openlifu.geo import (
    cartesian_to_spherical_vectorized,
    spherical_coordinate_basis,
    spherical_to_cartesian,
    spherical_to_cartesian_vectorized,
)
from openlifu.seg.skinseg import (
    compute_foreground_mask,
    create_closed_surface_from_labelmap,
    spherical_interpolator_from_mesh,
    vtk_img_from_array_and_affine,
)
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion

log = logging.getLogger("VirtualFit")

ras2asl_3x3 = np.array([[0,1,0],[0,0,1],[-1,0,0]], dtype=float) # ASL means Anterior-Superior-Left coordinates
asl2ras_3x3 = ras2asl_3x3.transpose()

@dataclass
class VirtualFitOptions(DictMixin):
    """Parameters to configure the `virtual_fit` algorithm.

    The terms 'pitch' and 'yaw' used here refer to the following target-centric angular coordinates in patient space:
        pitch: The angle between the anterior axis through the target and the ray from the target to the projection of
            a given point into the anterior-superior plane.
        yaw: The angle between the anterior-superior plane through the target and the ray from the target to a given point.

    Another way to describe them in terms of standard spherical coordinates centered at the target in ASL (anterior-superior-left) space:
        pitch: The azimuthal spherical coordinate.
        yaw: 90 degrees minus the polar spherical coordinate.
    """

    units: Annotated[str, OpenLIFUFieldData("Length units", "The units of length used in the length attributes of this class")] = "mm"
    """The units of length used in the length attributes of this class"""

    transducer_steering_center_distance: Annotated[float, OpenLIFUFieldData("Steering center distance", "Distance from the transducer origin axially to the center of the steering zone in the units `units`")] = 50.
    """Distance from the transducer origin axially to the center of the steering zone in the units `units`"""

    steering_limits: Annotated[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                               OpenLIFUFieldData("Steering limits", "Steering bounds along each axis from the transducer origin, in the units `units`")] = ((-50, 50), (-50, 50), (-50, 50))
    """Distance from the transducer origin axially to the center of the steering zone in the units `units`"""

    pitch_range: Annotated[Tuple[float, float], OpenLIFUFieldData("Pitch range (deg)", "Range of pitches to include in the transducer fitting search grid, in degrees")] = (-10, 150)
    """Range of pitches to include in the transducer fitting search grid, in degrees"""

    pitch_step: Annotated[float, OpenLIFUFieldData("Pitch step size (deg)", "Pitch step size when forming the transducer fitting search grid, in degrees")] = 5
    """Pitch step size when forming the transducer fitting search grid, in degrees"""

    yaw_range: Annotated[Tuple[float, float], OpenLIFUFieldData("Yaw range (deg)", "Range of yaws to include in the transducer fitting search grid, in degrees")] = (-65, 65)
    """Range of yaws to include in the transducer fitting search grid, in degrees"""

    yaw_step: Annotated[float, OpenLIFUFieldData("Yaw step size (deg)", "Yaw step size when forming the transducer fitting search grid, in degrees")] = 5
    """Yaw step size when forming the transducer fitting search grid, in degrees"""

    planefit_dyaw_extent: Annotated[float, OpenLIFUFieldData("Plane fit yaw extent", "Left and right extents of the point grid to be used for plane fitting along the local yaw axes, in units of `units`")] = 15
    """Left and right extents of the point grid to be used for plane fitting along the local yaw axes,
    in units of `units`. The plane fitting point grid will be twice this size, since this is left
    and right extents. (Note that this has units of length, not angle!)"""

    planefit_dyaw_step: Annotated[float, OpenLIFUFieldData("Plane fit yaw step", "Local yaw axis step size to use when constructing plane fitting grids. In spatial units of `units`")] = 3
    """Local yaw axis step size to use when constructing plane fitting grids. In spatial units of `units`."""

    planefit_dpitch_extent: Annotated[float, OpenLIFUFieldData("Plane fit pitch extent", "Left and right extents of the point grid to be used for plane fitting along the local pitch axes, in spatial units of `units`")] = 15
    """Left and right extents of the point grid to be used for plane fitting along the local pitch axes,
    in spatial units of `units`. The plane fitting point grid will be twice this size, since this is left
    and right extents."""

    planefit_dpitch_step: Annotated[float, OpenLIFUFieldData("Plane fit pitch step", "Local pitch axis step size to use when constructing plane fitting grids. In spatial units of `units`")] = 3
    """Local pitch axis step size to use when constructing plane fitting grids. In spatial units of `units`."""

    def to_units(self, target_units: str) -> VirtualFitOptions:
        """Do unit conversion and return a version of this VirtualFitOptions that uses
        `target_units` as the units for all attributes that have units of length."""
        conversion_factor = getunitconversion(from_unit = self.units, to_unit=target_units)
        return VirtualFitOptions(
            units = target_units,
            transducer_steering_center_distance = conversion_factor * self.transducer_steering_center_distance,
            steering_limits = tuple(map(tuple,conversion_factor*np.array(self.steering_limits))),
            pitch_range = self.pitch_range,
            pitch_step = self.pitch_step,
            yaw_range = self.yaw_range,
            yaw_step = self.yaw_step,
            planefit_dyaw_extent = conversion_factor * self.planefit_dyaw_extent,
            planefit_dyaw_step = conversion_factor * self.planefit_dyaw_step,
            planefit_dpitch_extent = conversion_factor * self.planefit_dpitch_extent,
            planefit_dpitch_step = conversion_factor * self.planefit_dpitch_step,
        )

    @staticmethod
    def from_dict(parameter_dict: Dict[str,Any]) -> VirtualFitOptions: # Override DictMixin here
        parameter_dict["pitch_range"] = tuple(parameter_dict["pitch_range"])
        parameter_dict["yaw_range"] = tuple(parameter_dict["yaw_range"])
        parameter_dict["steering_limits"] = tuple(map(tuple,parameter_dict["steering_limits"]))
        return VirtualFitOptions(**parameter_dict)


def virtual_fit(
    volume_array : np.ndarray,
    volume_affine_RAS : np.ndarray,
    units: str,
    target_RAS : Sequence[float],
    standoff_transform : np.ndarray,
    options : VirtualFitOptions,
) -> List[np.ndarray]:
    """Run patient-specific "virtual fitting" algorithm, suggesting a series of candidate transducer
    transforms for optimal sonicaiton of a given target.

    Args:
        volume_array: A 3D volume MRI
        volume_affine_RAS: A 4x4 affine transform that maps `volume_array` into RAS space with certain units
        units: The spatial units of the RAS space into which volume_affine_RAS maps
        target_RAS: A 3D point, in the coordinates and units of `volume_affine_RAS` (the `units` argument)
        standoff_transform: See the documentation of `create_standoff_transform` or
            `Transducer.standoff_transform` for the meaning of this. Here it should be provided in the
            units `units`. The method `Transducer.get_standoff_transform_in_units` is useful for getting this.
        options : Virtual fitting algorithm configuration. See the `VirtualFitOptions` documentation.

    Returns: A list of transducer transform candidates sorted starting from the best-scoring one. The transforms map transducer space
        into LPS space, and they are in the same units as the RAS space of `volume_affine_RAS` (aka the `units` argument).
    """

    # Express all virtual fit options in the units of volume_affine_RAS, i.e. the physical space of the volume
    options = options.to_units(units)
    pitch_range = options.pitch_range
    pitch_step = options.pitch_step
    yaw_range = options.yaw_range
    yaw_step = options.yaw_step
    transducer_steering_center_distance = options.transducer_steering_center_distance
    steering_limits = options.steering_limits
    planefit_dyaw_extent = options.planefit_dyaw_extent
    planefit_dyaw_step = options.planefit_dyaw_step
    planefit_dpitch_extent = options.planefit_dpitch_extent
    planefit_dpitch_step = options.planefit_dpitch_step

    log.info("Computing foreground mask...")
    foreground_mask_array = compute_foreground_mask(volume_array)
    foreground_mask_vtk_image = vtk_img_from_array_and_affine(foreground_mask_array, volume_affine_RAS)
    log.info("Creating closed surface from labelmap...")
    skin_mesh = create_closed_surface_from_labelmap(foreground_mask_vtk_image)
    log.info("Building skin interpolator...")
    skin_interpolator = spherical_interpolator_from_mesh(
        surface_mesh = skin_mesh,
        origin = target_RAS,
        xyz_direction_columns = asl2ras_3x3, # surface mesh was in RAS, so here spherical coordinates are placed on ASL space
    )

    # Useful transforms to and from the skin_interpolator ASL space and between RAS and LPS
    # Note that ASL is a left-handed coordinate system while RAS and LPS are right-handed.
    interpolator2ras = np.eye(4)
    interpolator2ras[:3,:3] = asl2ras_3x3
    interpolator2ras[:3,3] = target_RAS
    ras_lps_swap = np.diag([-1.,-1,1,1])
    interpolator2lps = ras_lps_swap @ interpolator2ras

    # Useful arrays for vectorized comparisons
    steering_mins = np.array([sl[0] for sl in steering_limits], dtype=float) # shape (3,). It is the lat,ele,ax steering min
    steering_maxs = np.array([sl[1] for sl in steering_limits], dtype=float) # shape (3,). It is the lat,ele,ax steering max

    log.info("Searching through candidate transducer poses...")

    # Construct search grid
    theta_sequence = np.arange(90 - yaw_range[-1], 90 - yaw_range[0], yaw_step)
    phi_sequence = np.arange(pitch_range[0], pitch_range[-1], pitch_step)
    theta_grid, phi_grid = np.meshgrid(theta_sequence, phi_sequence, indexing="ij") # each has shape (number of thetas, number of phis)
    num_thetas, num_phis = theta_grid.shape
    num_search_points = num_thetas*num_phis
    thetas = theta_grid.reshape(num_search_points)
    phis = phi_grid.reshape(num_search_points)

    transducer_poses = np.empty((num_search_points,4,4), dtype=float)
    in_bounds = np.zeros(shape=num_search_points, dtype=bool)
    steering_dists = np.zeros(shape=num_search_points, dtype=float)

    for i in range(num_search_points):
        theta_rad, phi_rad = thetas[i]*np.pi/180, phis[i]*np.pi/180

        # Cartesian coordinate location of the point at which we are fitting a plane
        point = np.array(spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad))

        # Build plane fitting grid in the spherical coordinate basis theta-phi plane, which we will later project back onto the skin surface
        dtheta_sequence = np.arange(-planefit_dyaw_extent, planefit_dyaw_extent + planefit_dyaw_step, planefit_dyaw_step)
        dphi_sequence = np.arange(-planefit_dpitch_extent, planefit_dpitch_extent + planefit_dpitch_step, planefit_dpitch_step)
        dtheta_grid, dphi_grid = np.meshgrid(dtheta_sequence, dphi_sequence, indexing='ij')

        r_hat, theta_hat, phi_hat = spherical_coordinate_basis(theta_rad,phi_rad)
        planefit_points_unprojected_cartesian = (
            point.reshape((1,1,3))
            + dtheta_grid[...,np.newaxis] * theta_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
            + dphi_grid[...,np.newaxis] * phi_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
        ) # shape (num dthetas, num dphis, 3)

        planefit_points_unprojected_spherical = cartesian_to_spherical_vectorized(
            planefit_points_unprojected_cartesian
        ) # shape (num dthetas, num dphis, 3)
        skin_projected_r_values = skin_interpolator(planefit_points_unprojected_spherical[...,1:]) # shape (num dthetas, num dphis) # TODO adjust docstrings to demand a *vectorizable* spherical interpolator
        planefit_points_cartesian = spherical_to_cartesian_vectorized( # Could instead renormalize planefit_points_unprojected_cartesian, not sure if it would give a speedup versus this
            np.stack([
                skin_projected_r_values, # New r values after projection to skin
                planefit_points_unprojected_spherical[...,1], # Same old theta values
                planefit_points_unprojected_spherical[...,2], # Same old phi values
            ], axis=-1)
        )

        # Fit the best plane to these points among the planes that pass through `point`. Here we find the normal vector to the plane.
        plane_normal = np.linalg.svd(
            planefit_points_cartesian.reshape(-1,3)-point.reshape(1,3),
            full_matrices=False, # we don't need the left-singular vectors anyway, so this speeds things up
        ).Vh[-1] # The right-singular vector corresponding to the smallest singular value


        # Transducer axial axis: Parallel to plane_normal, but points towards rather than away from the origin.
        plane_normal_norm = np.linalg.norm(plane_normal)
        if plane_normal_norm < 1e-10:
            continue # Bad geometry at this location, so it's not a virtual fit candidate
        transducer_z = - np.sign(np.dot(plane_normal,point)) * plane_normal / plane_normal_norm

        # Transducer elevational axis: Phi-hat, but then with its component along transducer_z eliminated. This orients the transducer "up" if this were forehead, for example.
        transducer_y = phi_hat - np.dot(phi_hat, transducer_z) * transducer_z
        transducer_y_norm = np.linalg.norm(transducer_y)
        if transducer_y_norm < 1e-10:
            continue # Bad geometry at this location, so it's not a virtual fit candidate
        transducer_y = transducer_y / transducer_y_norm

        # Transducer lateral axis, here simply the only remaining choice to keep it a left handed coordinate system
        # (ASL is left-handed, so the transducer axes must be left-handed to make for an orientation-preserving transducer transform)
        transducer_x = np.cross(transducer_z, transducer_y)

        transducer_transform = np.array(
            [
                [*transducer_x, 0],
                [*transducer_y, 0],
                [*transducer_z, 0],
                [*point, 1],
            ],
            dtype=float
        ).transpose()

        # The transform moves the transducer into the ASL skin interpolator space.
        # We want a transform that moves the transducer into LPS space, and we also want to apply the standoff transform
        transducer_transform = interpolator2lps @ transducer_transform @ standoff_transform

        # Target in transducer coordinates (lat, ele, ax)
        target_XYZ = (np.linalg.inv(transducer_transform) @ ras_lps_swap @ np.array([*target_RAS,1.0]))[:3]

        # Target in "steering space", where the origin is the center of the steering zone.
        target_steering_space = target_XYZ - np.array([0.,0.,transducer_steering_center_distance])

        steering_distance : float = float(np.linalg.norm(target_steering_space))

        # Check whether the target is in the steering range
        target_in_bounds : bool = bool(np.all((steering_mins < target_steering_space) & (target_steering_space < steering_maxs)))

        # Finally, fill out the arrays we have been building in this loop
        transducer_poses[i] = transducer_transform
        steering_dists[i] = steering_distance
        in_bounds[i] = target_in_bounds


    sorted_transforms = [
        x[0] for x in sorted(zip(transducer_poses[in_bounds],steering_dists[in_bounds]), key = lambda x : x[1])
    ]

    log.info("Virtual fitting complete.")

    return sorted_transforms
