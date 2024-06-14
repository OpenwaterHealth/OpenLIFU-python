import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Pulse:
    """
    Class for representing a sinusoidal pulse
    
    :ivar frequency: Frequency of the pulse in Hz
    :ivar amplitude: Amplitude of the pulse in Pa
    :ivar duration: Duration of the pulse in s
    """

    frequency: float = 1.0 # Hz
    amplitude: float = 1.0 # Pa
    duration: float = 1.0 # s

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
    
    def to_dict(self):
        """
        Convert the pulse to a dictionary

        :returns: Dictionary of the pulse parameters
        """
        return {"frequency": self.frequency,
                "amplitude": self.amplitude,
                "duration": self.duration,
                "class": "Pulse"}
    
    @staticmethod
    def from_dict(d):
        """
        Create a pulse from a dictionary

        :param d: Dictionary of the pulse parameters
        :returns: Pulse object
        """
        return Pulse(frequency=d["frequency"], amplitude=d["amplitude"], duration=d["duration"])
    