from dataclasses import dataclass

import numpy as np
import xarray as xa

from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class VirtualFit:
    search_range: tuple[float, float] = (-10., 10.) #pitch yaw
    """min-max position to search"""

    search_range_units: str = "deg"
    """min-max position units to search"""

    def extract_skin_surface(self, volume: xa.Dataset):
        # from scipy.spatial import ConvexHull
        # ConvexHull(volume)
        pass

    def fit_to_skin(self):
        pass

    def get_search_grid(self, skin_surface: np.ndarray):
        """
        defines the transducer search grid
        each grid element is a 4x4 matrix that correctly
        transform the Transducer position so it fit to skin surface.
        """
        # fit_to_skin()
        # self.search_range
        pass

    def analyse_position(self, pos: np.ndarray, transducer: Transducer, target: Point):
        """Minimal implementation is closest to target"""
        pass

    def run(self, volume: xa.Dataset=None, transducer: Transducer=None, target: Point=None):
        # 1. extract_skin_surface
        # 2. get_search_grid(skin_surface)
        # 3. [analyse_position(pos, transducer, target) for pos in self.search_grid]
        pass
