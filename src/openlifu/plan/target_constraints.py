from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin


@dataclass
class TargetConstraints(DictMixin):
    """A class for storing target constraints.

    Target constraints are used to define the acceptable range of
    positions for a target. For example, a target constraint could
    be used to define the acceptable range of values for the x position
    of a target.
    """

    dim: Annotated[str, OpenLIFUFieldData("Constrained dimension ID", "The dimension ID being constrained")] = "x"
    """The dimension ID being constrained"""

    name: Annotated[str, OpenLIFUFieldData("Constrained dimension name", "The name of the dimension being constrained")] = "dim"
    """The name of the dimension being constrained"""

    units: Annotated[str, OpenLIFUFieldData("Dimension units", "The units of the dimension being constrained")] = "m"
    """The units of the dimension being constrained"""

    min: Annotated[float, OpenLIFUFieldData("Minimum allowed value", "The minimum value of the dimension")] = float("-inf")
    """The minimum value of the dimension"""

    max: Annotated[float, OpenLIFUFieldData("Maximum allowed value", "The maximum value of the dimension")] = float("inf")
    """The maximum value of the dimension"""

    def check_bounds(self, pos: float):
        """Check if the given position is within bounds."""

        if (pos < self.min) or (pos > self.max):
            logging.error(msg=f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")
            raise ValueError(f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")
