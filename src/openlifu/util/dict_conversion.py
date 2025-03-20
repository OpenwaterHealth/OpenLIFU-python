from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Type, TypeVar, get_origin

import numpy as np

T = TypeVar('T', bound='DictMixin')

@dataclass
class DictMixin:
    """Mixin for basic conversion of a dataclass to and from dict."""
    def to_dict(self) -> Dict[str,Any]:
        """
        Convert the object to a dictionary

        Returns: Dictionary of object parameters
        """
        return asdict(self)

    @classmethod
    def from_dict(cls : Type[T], parameter_dict:Dict[str,Any]) -> T:
        """
        Create an object from a dictionary

        Args:
            parameter_dict: dictionary of parameters to define the object
        Returns: new object
        """
        if "class" in parameter_dict:
            parameter_dict.pop("class")
        new_object = cls(**parameter_dict)

        # Convert anything that should be a numpy array to numpy
        for field in fields(cls):
            # Note that sometimes "field.type" is a string rather than a type due to the "from annotations import __future__" stuff
            if get_origin(field.type) is np.ndarray or field.type is np.ndarray or "np.ndarray" in field.type:
                setattr(new_object, field.name, np.array(getattr(new_object,field.name)))

        return new_object
