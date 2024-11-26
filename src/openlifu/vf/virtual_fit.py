import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
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

    steering_limits: Tuple[TargetConstraints] = field(default_factory=list)
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
        """The VirtualFit logger."""
        self.logger.info(f"Initializing VirtualFit with the following parameters: {self.__dict__}")
        self.logger.info("VirtualFit: Skin extraction...")
        # 1. extract skin surface, this is done only once at initialization
        # self.skin_origin, self.skin_surface, self.skin_interpolator = self.extract_skin_surface(volume: xa.Dataset)
        """A list of vertices representing the skin surface."""

    def extract_skin_surface(
            self,
            volume: xa.Dataset,
            quantile: float = 0.05,
            scene_origin: Optional[Tuple[float, float, float]] = None,
            scene_matrix: Optional[np.ndarray] = None):
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

    def fit_to_surface(
            self,
            sph_coords: Tuple[float, float],
            skin_surface: np.ndarray
        ) -> np.ndarray:
        """
        Fit a 3D plane plane given spherical coordinates (yaw, pitch)
        and a set of points coordinates LPS.
        """
        pass

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

    def analyse_position(self, pos: np.ndarray, transducer: Transducer, target: Point):
        """
        Analyse the transducer position relative to a specific target.
        """
        #TODO: Compute if target is within steering limits
        #TODO: In the future, we should implement the ray-tracing analysis given a full segmentation

        # pos_analysis = 1.0
        # target_tr_space = target2trspace(pos, target)
        # for target_constraint in self.steering_limits:
        #     pos = target_tr_space.get_position(
        #         dim=target_constraint.dim,
        #         units=target_constraint.units
        #     )
        #     try:
        #         target_constraint.check_bounds(pos)
        #     except ValueError:
        #         pos_analysis = 0.0
        #
        # return pos_analysis

        pass

    def run(
            self,
            target: Point,
            pitch_range: Optional[Tuple[int, int]] = None,
            pitch_step: Optional[int] = None,
            yaw_range: Optional[Tuple[int, int]] = None,
            yaw_step: Optional[int] = None,
            steering_limits: Optional[Tuple[TargetConstraints]] = None,
            blocked_elems_threshold: Optional[float] = None
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
        if steering_limits is None:
            steering_limits = self.steering_limits
        if blocked_elems_threshold is None:
            blocked_elems_threshold = self.blocked_elems_threshold

        self.logger.info("Running VirtualFit main process.")
        self.logger.info("VirtualFit: Searching optimal position...")
        # 2. get search grid
        search_grid = self.get_search_grid(yaw_range, yaw_step, pitch_range, pitch_step)
        for i in range(search_grid[0].shape[0]):
            for j in range(search_grid[0].shape[1]):
                yaw, pitch = (search_grid[0][i, j], search_grid[1][i, j])
                self.logger.info(f"VirtualFit: Analysing {(yaw, pitch)}...")
                # 3. define transducer transform (plane fitting) on the surface (skin) given spherical coordinate (yaw, pitch)
                # self.fit_to_surface(sph_coords: Tuple[float, float], skin_surface: np.ndarray)
                # 4. analyse current transform
                # self.analyse_position(pos: np.ndarray, transducer: Transducer, target: Point)
                optimal_transform = np.zeros((4, 4))
        self.logger.info("VirtualFit: Found optimal position!")

        return optimal_transform
