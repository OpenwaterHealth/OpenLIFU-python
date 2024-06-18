from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import xarray as xa
from pyfus.xdc import Transducer
from pyfus.geo import Point
from pyfus.bf import delay_methods

@dataclass
class DelayMethod(ABC):
    @abstractmethod
    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: bool = True):
        pass

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d
    
    @staticmethod
    def from_dict(d):
        d = d.copy()
        short_classname = d.pop("class")
        module_dict = delay_methods.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)