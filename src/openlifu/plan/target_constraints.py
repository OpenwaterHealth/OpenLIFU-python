import logging
from dataclasses import dataclass

from openlifu.io.dict_conversion import DictMixin


@dataclass
class TargetConstraints(DictMixin):
    """ A class for storing target constraints.

    Target constraints are used to define the acceptable range of
    positions for a target. For example, a target constraint could
    be used to define the acceptable range of values for the x position
    of a target.
    """

    dim: str = "x"
    """The dimension ID being constrained"""

    name: str = "dim"
    """The name of the dimension being constrained"""

    units: str = "m"
    """The units of the dimension being constrained"""

    min: float = float("-inf")
    """The minimum value of the dimension"""

    max: float = float("inf")
    """The maximum value of the dimension"""

    def check_bounds(self, pos: float):
        """Check if the given position is within bounds."""

        if (pos < self.min) or (pos > self.max):
            logging.error(msg=f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")
            raise ValueError(f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")
