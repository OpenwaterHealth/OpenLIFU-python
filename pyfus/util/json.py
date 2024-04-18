import json
import os
import numpy as np
from pyfus.geo import Point
from pyfus.xdc.transducer import Transducer
from pyfus.xdc.element import Element
from pyfus.seg.material import Material
from pyfus.db.session import Session
from pyfus.db.subject import Subject

class PYFUSEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Point):
            return obj.to_dict()
        if isinstance(obj, Transducer):
            return obj.to_dict()
        if isinstance(obj, Element):
            return obj.to_dict()
        if isinstance(obj, Material):
            return obj.to_dict()
        if isinstance(obj, Session):
            return obj.to_dict()
        if isinstance(obj, Subject):
            return obj.to_dict()
        return super(PYFUSEncoder, self).default(obj)

def to_json(obj, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'w') as file:
        json.dump(obj, file, cls=PYFUSEncoder, indent=4)    