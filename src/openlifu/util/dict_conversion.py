from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

T = TypeVar('T', bound='DictMixin')

@dataclass
class DictMixin:
    """Mixin for basic conversion to and from dict."""
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
        return cls(**parameter_dict)
