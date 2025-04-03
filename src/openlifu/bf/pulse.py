from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin


@dataclass
class Pulse(DictMixin):
    """
    Class for representing a sinusoidal pulse
    """

    frequency: Annotated[float, OpenLIFUFieldData("Frequency (Hz)", "Frequency of the pulse in Hz")] = 1.0  # Hz
    """Frequency of the pulse in Hz"""

    amplitude: Annotated[float, OpenLIFUFieldData("Amplitude (V)", "Amplitude of the pulse in volts")] = 1.0  # V
    """Amplitude of the pulse in volts"""

    duration: Annotated[float, OpenLIFUFieldData("Duration (s)", "Duration of the pulse in s")] = 1.0  # s
    """Duration of the pulse in s"""

    def calc_pulse(self, t: np.array):
        """
        Calculate the pulse at the given times

        :param t: Array of times to calculate the pulse at (s)
        :returns: Array of pulse values at the given times
        """
        return self.amplitude * np.sin(2*np.pi*self.frequency*t)

    def calc_time(self, dt: float):
        """
        Calculate the time array for the pulse for a particular timestep

        :param dt: Time step (s)
        :returns: Array of times for the pulse (s)
        """
        return np.arange(0, self.duration, dt)

    def get_table(self):
        """
        Get a table of the pulse parameters

        :returns: Pandas DataFrame of the pulse parameters
        """
        records = [{"Name": "Frequency", "Value": self.frequency, "Unit": "Hz"},
                   {"Name": "Amplitude", "Value": self.amplitude, "Unit": "Pa"},
                   {"Name": "Duration", "Value": self.duration, "Unit": "s"}]
        return pd.DataFrame.from_records(records)
