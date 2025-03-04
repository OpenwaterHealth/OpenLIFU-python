from dataclasses import dataclass

import pandas as pd

from openlifu.util.dict_conversion import DictMixin


@dataclass
class Sequence(DictMixin):
    """
    Class for representing a sequence of pulses

    :ivar pulse_interval: Interval between pulses in the sequence (s)
    :ivar pulse_count: Number of pulses in the sequence
    :ivar pulse_train_interval: Interval between pulse trains in the sequence (s)
    :ivar pulse_train_count: Number of pulse trains in the sequence
    """
    pulse_interval: float = 1.0 # s
    pulse_count: int = 1
    pulse_train_interval: float = 1.0 # s
    pulse_train_count: int = 1

    def __post_init__(self):
        if self.pulse_interval <= 0:
            raise ValueError("Pulse interval must be positive")
        if self.pulse_count <= 0:
            raise ValueError("Pulse count must be positive")
        if self.pulse_train_interval < 0:
            raise ValueError("Pulse train interval must be non-negative")
        elif (self.pulse_train_interval > 0) and (self.pulse_train_interval < (self.pulse_interval * self.pulse_count)):
            raise ValueError("Pulse train interval must be greater than or equal to the total pulse interval")
        if self.pulse_train_count <= 0:
            raise ValueError("Pulse train count must be positive")

    def get_table(self) -> pd.DataFrame:
        """
        Get a table of the sequence parameters

        :returns: Pandas DataFrame of the sequence parameters
        """
        records = [
            {"Name": "Pulse Interval", "Value": self.pulse_interval, "Unit": "s"},
            {"Name": "Pulse Count", "Value": self.pulse_count, "Unit": ""},
            {"Name": "Pulse Train Interval", "Value": self.pulse_train_interval, "Unit": "s"},
            {"Name": "Pulse Train Count", "Value": self.pulse_train_count, "Unit": ""}
        ]
        return pd.DataFrame.from_records(records)
