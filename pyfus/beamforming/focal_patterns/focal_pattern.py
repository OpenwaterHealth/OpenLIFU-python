from abc import ABC, abstractmethod
from dataclasses import dataclass
from pyfus.geo import Point
from pyfus.beamforming import focal_patterns

@dataclass
class FocalPattern(ABC):
    """
    Abstract base class for representing a focal pattern

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    """
    target_pressure: float = 1.0 # Pa

    @abstractmethod
    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern
        
        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        pass

    @abstractmethod
    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci
        """
        pass

    def to_dict(self):
        """
        Convert the focal pattern to a dictionary

        :returns: Dictionary of the focal pattern parameters
        """
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        """
        Create a focal pattern from a dictionary
        
        :param d: Dictionary of the focal pattern parameters
        :returns: FocalPattern object
        """
        d = d.copy()        
        short_classname = d.pop("class")
        module_dict = focal_patterns.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)