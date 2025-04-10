from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np

from openlifu.bf.focal_patterns import FocalPattern
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class Wheel(FocalPattern):
    """
    Class for representing a wheel pattern
    """

    center: Annotated[bool, OpenLIFUFieldData("Include center point?", "Whether to include the center for the wheel pattern")] = True
    """Whether to include the center for the wheel pattern"""

    num_spokes: Annotated[int, OpenLIFUFieldData("Number of spokes", "Number of spokes in the wheel pattern")] = 4
    """Number of spokes in the wheel pattern"""

    spoke_radius: Annotated[float, OpenLIFUFieldData("Spoke radius", "Radius of the spokes in the wheel pattern")] = 1.0  # mm
    """Radius of the spokes in the wheel pattern"""

    units: Annotated[str, OpenLIFUFieldData("Units", "Units of the wheel pattern parameters")] = "mm"
    """Units of the wheel pattern parameters"""

    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        if self.center:
            targets = [target.copy()]
            targets[0].id = f"{target.id}_center"
            targets[0].id = f"{target.id} (Center)"
        else:
            targets = []
        m = target.get_matrix(center_on_point=True)
        for i in range(self.num_spokes):
            theta = 2*np.pi*i/self.num_spokes
            local_position = self.spoke_radius * np.array([np.cos(theta), np.sin(theta), 0.0])
            position = np.dot(m, np.append(local_position, 1.0))[:3]
            spoke = Point(id=f"{target.id}_{np.rad2deg(theta):.0f}deg",
                              name=f"{target.name} ({np.rad2deg(theta):.0f}°)",
                              position=position,
                              units=self.units,
                              radius=target.radius)
            targets.append(spoke)
        return targets

    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci
        """
        return int(self.center) + self.num_spokes
