from __future__ import annotations

import logging
from dataclasses import dataclass

from openlifu.util.dict_conversion import DictMixin


@dataclass
class TargetConstraint(DictMixin):
    """ A class for storing target constraints.

    Target constraints are used to define the acceptable range of
    positions for a target. For example, a target constraint could
    be used to define the acceptable range of values for the x position
    of a target.
    """
    min: float|None = None
    """The minimum value of the dimension"""

    max: float|None = None
    """The maximum value of the dimension"""

    units: str = "mm"
    """The units of the dimension being constrained"""

    def is_in_bounds(self, pos:float):
        """Check if the given position is within bounds."""
        if self.min is not None and pos < self.min:
            return False
        if self.max is not None and pos > self.max:
            return False
        return True

    def check_bounds(self, pos: float):
        """Check if the given position is within bounds."""
        if not self.is_in_bounds(pos):
            logging.error(msg=f"The position {pos} is not within bounds [{self.min}, {self.max}]!")
            raise ValueError(f"The position {pos} is not within bounds [{self.min}, {self.max}]!")
