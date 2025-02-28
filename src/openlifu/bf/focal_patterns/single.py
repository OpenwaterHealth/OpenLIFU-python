from __future__ import annotations

from dataclasses import dataclass

from openlifu.bf.focal_patterns import FocalPattern
from openlifu.geo import Point


@dataclass
class SinglePoint(FocalPattern):
    """
    Class for representing a single focus

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    """
    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        return [target.copy()]

    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci (1)
        """
        return 1
