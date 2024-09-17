"""Helper utilities for unit tests"""

from dataclasses import fields, is_dataclass

import numpy as np


def dataclasses_are_equal(obj1, obj2) -> bool:
    """Return whether two nested dataclass structures are equal by recursively checking for equality of fields,
    while specially handling numpy arrays.

    Recurses into dataclasses as well as dictionary-like, list-like, and tuple-like fields.
    """
    obj_type = type(obj1)
    if type(obj2) != obj_type:
        return False
    elif is_dataclass(obj_type):
        return all(
            dataclasses_are_equal(getattr(obj1, f.name), getattr(obj2, f.name))
            for f in fields(obj_type)
        )
    # handle the builtin types first for speed; subclasses handled below
    elif issubclass(obj_type, list) or issubclass(obj_type, tuple):
        return all(dataclasses_are_equal(v1,v2) for v1,v2 in zip(obj1,obj2))
    elif issubclass(obj_type, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(dataclasses_are_equal(obj1[k],obj2[k]) for k in obj1)
    elif issubclass(obj_type, np.ndarray):
        return (obj1==obj2).all()
    else:
        return obj1 == obj2
