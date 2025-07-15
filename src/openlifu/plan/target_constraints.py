from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunittype


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

    def __post_init__(self):
        if not isinstance(self.dim, str):
            raise TypeError("Dimension ID must be a string")
        if not isinstance(self.name, str):
            raise TypeError("Dimension name must be a string")
        if not isinstance(self.units, str):
            raise TypeError("Dimension units must be a string")
        if getunittype(self.units) != 'distance':
            raise ValueError(f"Units must be a length unit, got {self.units}")
        if not isinstance(self.min, (int, float)):
            raise TypeError("Minimum value must be a number")
        if not isinstance(self.max, (int, float)):
            raise TypeError("Maximum value must be a number")
        if self.min > self.max:
            raise ValueError("Minimum value cannot be greater than maximum value")

    def check_bounds(self, pos: float):
        """Check if the given position is within bounds."""

        if (pos < self.min) or (pos > self.max):
            logging.error(msg=f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")
            raise ValueError(f"The position {pos} at dimension {self.name} is not within bounds [{self.min}, {self.max}]!")

    def to_table(self):
        """
        Get a table of the target constraints parameters.
        :returns: Pandas DataFrame of the target constraints parameters
        """
        import pandas as pd
        records = [
            {"Name": self.name, "Value": f"{self.min} - {self.max}", "Unit": self.units},
        ]
        return pd.DataFrame.from_records(records)
