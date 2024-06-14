from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import xarray as xa
from openlifu.xdc import Transducer
from openlifu.geo import Point
from openlifu.bf import apod_methods

@dataclass
class ApodizationMethod(ABC):
    @abstractmethod
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform: bool = True):
        pass

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        d = d.copy()
        short_classname = d.pop("class")
        module_dict = apod_methods.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)
