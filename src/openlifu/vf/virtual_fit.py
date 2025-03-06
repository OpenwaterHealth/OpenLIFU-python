from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import scipy.interpolate
import xarray as xa

from openlifu.db.session import ArrayTransform
from openlifu.geo import Point
from openlifu.plan import TargetConstraints
from openlifu.xdc import Transducer


@dataclass
class VirtualFit:
    """
    VirtualFit class.

    Represents the virtual fitting algorithm which consists in
    finding the optimal transducer transform (position and orientation)
    given an input MRI volume in LPS coordinates and the associated target.
    """
    pitch_range: Tuple[int, int] = (-10, 40)
    """The pitch range for the grid search."""

    pitch_step: int = 3
    """The pitch step for the grid search."""

    yaw_range: Tuple[int, int] = (-5, 25)
    """The yaw range for the grid search."""

    yaw_step: int = 3
    """The yaw step for the grid search."""

    search_range_units: str = "deg"
    """Search grid units."""

    radius_in_mm: float = 50
    """Radius from transducer"""

    steering_limits: Tuple[TargetConstraints, TargetConstraints, TargetConstraints] = (TargetConstraints(), TargetConstraints(), TargetConstraints())
    """Defines the steering range limits for the transducer in the local coordinate system, usually in (lat, ele, ax)."""

    blocked_elems_threshold: float = 0.1
    """How much blocked elements are acceptable."""

    volume: xa.Dataset = field(default_factory=xa.Dataset)
    """The MRI volume in LPS coordinates, on which to optimize the position."""

    scene_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    """The transform represents the MRI volume scene"""

    scene_origin: Tuple[float, float, float] = (0, 0, 0)
    """The origin point of the the MRI volume scene"""

    transducer: Transducer = field(default_factory=Transducer)
    """Transducer that sits on the skin."""

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing VirtualFit with the following parameters: {self.__dict__}")
        self.logger.info("VirtualFit: Skin extraction...")
        # 1. extract skin surface, this is done only once at initialization
        # self.skin_origin, self.skin_surface, self.skin_interpolator = self.extract_skin_surface(volume: xa.Dataset)

    def extract_skin_surface(
            self,
            volume: xa.Dataset,
            quantile: float = 0.05,
            scene_origin: Tuple[float, float, float] | None = None,
            scene_matrix: np.ndarray | None = None):
        """
        Extract skin surface from MRI volume in LPS coordinates

        Args:
            volume: The MRI volume in LPS coordinates.
                Target is expected to be in the simulation grid coordinates (lat, ele, ax).
            quantile: The threshold to define the surface.
            scene_origin: The origin of the scene
            scene_matrix: The transform of the scene

        Returns:
            skin_origin: The origin of the skin
            skin_surface: The list of points represent the skin surface in LPS coordinates
            skin_interpolator: An interpolatoor represents the skin surface in spherical coordinates (pitch, yaw, r)
        """
        if scene_origin is None:
            scene_origin = self.scene_origin
        if scene_matrix is None:
            scene_matrix = self.scene_matrix
        # -> Tuple[float, float, float], np.ndarray, scipy.interpolate.LinearNDInterpolator]
        #TODO: Segmentation (basic thresholding)
        # threshold = np.quantile(volume, 0.05)   #TODO: check otsu threhsolding instead
        # volume_thresholded = volume[volume > threshold]

        #TODO: option1 Intepolant + list of points
        # from scipy.interpolate import LinearNDInterpolator
        # return Tuple[float, float, float], np.ndarray, skin_interpolator(scipy.interpolate.LinearNDInterpolator)

        #TODO: option2 ConvexHull (combine interpolant and list of points)
        # from scipy.spatial import ConvexHull
        # return Tuple[float, float, float], ConvexHull(volume)
        pass

    def pyr2lps(self, pitch: float, yaw: float, r: float, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        Convert spherical coordinates to LPS coordinates
        """
        pitch_rad = np.deg2rad(180 - pitch)
        yaw_rad = np.deg2rad(yaw)
        p = r * np.cos(yaw_rad) * np.cos(pitch_rad)
        s = r * np.cos(yaw_rad) * np.sin(pitch_rad)
        l = r * np.sin(yaw_rad)
        return l + origin[0], p + origin[1], s + origin[2]

    def lps2pyr(self, l: float, p:float, s:float, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        Convert LPS coordinates to spherical coordinates
        """
        x = p - origin[1]
        y = s - origin[2]
        z = l - origin[0]
        th = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        r = np.sqrt(x**2 + y**2 + z**2)
        pitch = (360 - np.degrees(th)) % 360 - 180
        yaw = np.degrees(phi)
        return pitch, yaw, r

    def get_transducer_pose(
            self,
            sph_coords: Tuple[float, float],
            skin_origin: Tuple[float, float, float] | None = None,
            skin_interpolator:  scipy.interpolate.LinearNDInterpolator | None = None,
            z_offset: float = 13.55,
            dzdy: float = 0.15,
            search_x: float = 20,
            search_dx: float = 1,
            search_y: float = 20,
            search_dy: float = 1) -> np.ndarray:
        """
        Computes the pose of the transducer positioned at a point on the segmented skin surface
        defined by spherical coordinates (pitch, yaw).

        Args:
            sph_coords: Spherical coordinates (pitch, yaw) in degrees.
                pitch: Angle above the S=0 "eye" line (rotation about the "L" axis).
                yaw: Angle along the pitched circle towards the subject's left ear
                    (rotation about the "S*" axis).
            skin_origin: The skin surface origin
            skin_interpolant: Function mapping spherical coordinates to radial distance.
            z_offset: Distance of transducer from skin surface (mm)
            dzdy: Slope of transducer away from skin surface. Default is 0.15 (bottom of transducer is raised 15% relative to top)
            search_x: Lateral (yaw) ROI extent for surface fitting (one-sided, mm).
            search_dx: Lateral (yaw) ROI step size for surface fitting (one-sided, mm)
            search_y: Elevation (pitch) ROI extent for surface fitting (one-sided, mm)
            search_dy: Elevation (pitch) ROI step size for surface fitting (one-sided, mm)

        Returns:
            np.ndarray
                4x4 transformation matrix representing the transducer's pose in terms of position and orientation (lat, ele, ax).
        """

        # Get input arguments
        pitch, yaw = sph_coords
        # Decomment these lines when the function extract_skin_surface is implemented
        # if skin_origin is None:
        #     skin_origin = self.skin_origin
        # if skin_interpolator is None:
        #     skin_interpolator = self.skin_interpolator

        # Compute skin surface origin and local coordinates
        r = skin_interpolator(pitch, yaw)
        l, p, s = self.pyr2lps(pitch, yaw, r, skin_origin)
        transducer_origin = np.array([l, p, s])

        # Set up local unit vectors for ROI definition
        roi_uv = [None] * 3
        roi_uv[2] = -transducer_origin / np.linalg.norm(transducer_origin, 2)
        l1, p1, s1 = self.pyr2lps(pitch, yaw - 1, r, skin_origin)
        roi_uv[0] = np.array([l1, p1, s1]) - transducer_origin
        roi_uv[0] -= roi_uv[2] * np.dot(roi_uv[0], roi_uv[2])
        roi_uv[0] /= np.linalg.norm(roi_uv[0], 2)
        roi_uv[1] = np.cross(roi_uv[2], roi_uv[0])
        # Create matrix
        roi_matrix = np.eye(4)
        roi_matrix[:3, :3] = np.column_stack(roi_uv)
        roi_matrix[:3, 3] = transducer_origin
        roi_forward_matrix = np.linalg.pinv(roi_matrix)

        # Search grid of transducer plane and surface fitting
        dx_sequence = np.arange(-search_x, search_x + search_dx, search_dx)
        dy_sequence = np.arange(-search_y, search_y + search_dy, search_dy)
        dx_grid, dy_grid = np.meshgrid(dx_sequence, dy_sequence, indexing='ij')
        roi_grid = np.array([l, p, s]) + np.outer(dx_grid, roi_uv[0]) + np.outer(dy_grid, roi_uv[1])
        # Convert search grid to pitch-yaw
        roi_pgrid = [self.lps2pyr(grid[0], grid[1], grid[2], skin_origin) for grid in roi_grid]
        # Get surface grid
        surf_pgrid = roi_pgrid.copy()
        surf_pgrid = [[*grid[:2], skin_interpolator(grid[0], grid[1]).item()] for grid in roi_pgrid]
        surf_lps = [self.pyr2lps(grid[0], grid[1], grid[2], skin_origin) for grid in surf_pgrid]
        # Get surface grid in local coords
        surf_lps_vec = np.hstack([surf_lps, np.ones((len(surf_lps), 1))]).T
        surf_xyz = roi_forward_matrix @ surf_lps_vec

        # Fit plane
        plane_fit = np.linalg.lstsq(surf_xyz[:2, :].T, surf_xyz[2, :], rcond=None)[0]
        # Get plane-fit unit vectors and convert to LPS
        plane_matrix_xyz = np.column_stack([[1, 0, plane_fit[0]], [0, 1, plane_fit[1]], [0, 0, 1]])
        plane_matrix_xyz /= np.linalg.norm(plane_matrix_xyz, axis=0)
        plane_matrix_xyz[:, 2] = np.cross(plane_matrix_xyz[:, 0], plane_matrix_xyz[:, 1])
        plane_matrix = np.eye(4)
        plane_matrix[:3, :3] = roi_matrix[:3, :3] @ plane_matrix_xyz
        plane_matrix[:3, 3] = transducer_origin

        # Get offset transducer unit vectors & origin
        transducer_origin = transducer_origin - plane_matrix[:3, 2] * z_offset + plane_matrix[:3, 1] * z_offset * dzdy
        transducer_uv = [None] * 3
        transducer_uv[0] = plane_matrix[:3, 0]
        transducer_uv[1] = plane_matrix[:3, 1] + dzdy * plane_matrix[:3, 2]
        transducer_uv[1] /= np.linalg.norm(transducer_uv[1], 2)
        transducer_uv[2] = np.cross(transducer_uv[0], transducer_uv[1])

        # Create matrix
        transducer_pose = np.eye(4)
        transducer_pose[:3, :3] = np.column_stack(transducer_uv)
        transducer_pose[:3, 3] = transducer_origin

        return transducer_pose

    def get_search_grid(
            self,
            yaw_range: Tuple[int, int],
            yaw_step: int,
            pitch_range: Tuple[int, int],
            pitch_step: int
        ) -> np.ndarray:
        """
        Defines the transducer search grid in (yaw, pitch) coordinates.
        """
        yaw_sequence = np.arange(yaw_range[0], yaw_range[-1], yaw_step)
        pitch_sequence = np.arange(pitch_range[0], pitch_range[-1], pitch_step)
        pitch_yaw_grid = np.meshgrid(pitch_sequence, yaw_sequence, indexing="ij")

        return pitch_yaw_grid

    def analyse_target_position(
            self,
            target: Point,
            transducer_pose: np.ndarray,
            radius_in_mm: float | None = None,
            steering_limits: Tuple[TargetConstraints, TargetConstraints, TargetConstraints] | None = None):
        """
        Analyzes the pose of a transducer relative to a specific target point.
        Determines whether or not the target is within the transducer's steering limits and
        computes the steering distance.

        Args:
            target: The target point
            transducer_pose : A 4x4 transformation matrix representing the transducer's pose
            radius_in_mm: Radius of the transducer in millimeters.
            steeringLimits: Steering range limits for the transducer in the local coordinate system (lat, ele, ax)

        Returns:
            in_bounds: A boolean indicating whether the target is within the steering limits.
            steering_dist: The Euclidean distance from the transducer's center to the target in the local coordinate system.
        """
        #TODO: In the future, we should implement the ray-tracing analysis given a full segmentation

        # Get transducer parameters
        if radius_in_mm is None:
            radius_in_mm = self.radius_in_mm
        if steering_limits is None:
            steering_limits = self.steering_limits


        # Transform target position into local coordinate of transducer
        homogeneous_target_position = np.append(target.position, 1)
        transducer_forward_matrix = np.linalg.pinv(transducer_pose)
        target_pos_local = transducer_forward_matrix @ homogeneous_target_position
        pos = target_pos_local[:3]
        pos[2] -= radius_in_mm

        # Calculate steering distance
        steering_dist = np.linalg.norm(pos)

        # Check if the target point is within the steering limits
        in_bounds = True
        for i, target_constraint in enumerate(steering_limits):
            try:
                target_constraint.check_bounds(pos[i])
            except ValueError:
                in_bounds = False

        return in_bounds, steering_dist

    def run(
            self,
            target: Point,
            pitch_range: Tuple[int, int] | None = None,
            pitch_step: int | None = None,
            yaw_range: Tuple[int, int] | None = None,
            yaw_step: int | None = None,
            radius_in_mm: float | None = None,
            steering_limits: Tuple[TargetConstraints] | None = None,
            blocked_elems_threshold: float | None = None
        ) -> ArrayTransform:
        """
        VirtualFit main process.

        Finds the optimal transducer transform (position and orientation)
        given an input MRI volume in LPS coordinates, and the associated
        target in same coordinates LPS.
        """
        if pitch_range is None:
            pitch_range = self.pitch_range
        if pitch_step is None:
            pitch_step = self.pitch_step
        if yaw_range is None:
            yaw_range = self.yaw_range
        if yaw_step is None:
            yaw_step = self.yaw_step
        if radius_in_mm is None:
            radius_in_mm = self.radius_in_mm
        if steering_limits is None:
            steering_limits = self.steering_limits
        if blocked_elems_threshold is None:
            blocked_elems_threshold = self.blocked_elems_threshold

        self.logger.info("Running VirtualFit main process.")
        self.logger.info("VirtualFit: Searching optimal position...")
        # 2. get search grid
        search_grid = self.get_search_grid(yaw_range, yaw_step, pitch_range, pitch_step)
        transducer_poses = np.empty(search_grid[0].shape, dtype=object)
        in_bounds = np.zeros_like(search_grid[0])
        steering_dists = np.zeros_like(search_grid[0])
        for i in range(search_grid[0].shape[0]):
            for j in range(search_grid[0].shape[1]):
                pitch, yaw = (search_grid[0][i, j], search_grid[1][i, j])
                self.logger.info(f"VirtualFit: Analysing {(pitch, yaw)}...")
                # 3. define transducer transform (plane fitting) on the surface (skin) given spherical coordinate (pitch, yaw)
                transducer_poses[i, j] = self.get_transducer_pose([pitch, yaw])
                # 4. analyse current transform
                in_bounds[i, j], steering_dists[i, j] = self.analyse_target_position(transducer_poses[i, j], target)
        # 5. get optimal transform
        optimal_transform = None
        for i in range(in_bounds.shape[0]):
            for j in range(in_bounds.shape[1]):
                if in_bounds[i, j]:
                    #TODO: Check blocked element
                    # self.check_blocked_elements()
                    optimal_transform = transducer_poses[i, j]
        self.logger.info("VirtualFit: Found optimal position!")

        return optimal_transform
