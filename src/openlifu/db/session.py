import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from openlifu.geo import Point
from openlifu.util.json import PYFUSEncoder
from openlifu.util.strings import sanitize


@dataclass
class ArrayTransform:
    """Class representing the transform on a transducer array to position it in space.

    matrix: 4x4 affine transform matrix
    units: the units of the space on which to apply the transform matrix , e.g. "mm"
        (In order to apply the transform to transducer points, first represent
        the points in these units.)
    """
    matrix: np.ndarray
    units : str

@dataclass
class Session:
    """
    Class representing a session

    ivar id: ID of the session
    ivar subject_id: ID of the subject
    ivar name: Name of the session
    date: Date of the session
    targets: sonication targets
    markers: registration markers
    volume_id: id of the subject volume file
    transducer_id: id of the transducer
    array_transform: transducer affine transform matrix with units
    attrs: Dictionary of attributes
    date_modified: Date of last modification
    """
    id: Optional[str] = None
    subject_id: Optional[str] = None
    name: Optional[str] = None
    date: datetime = datetime.now()
    targets: List[Point] = field(default_factory=list)
    markers: List[Point] = field(default_factory=list)
    volume_id: Optional[str] = None
    transducer_id: Optional[str] = None
    array_transform: ArrayTransform = field(default_factory=lambda : ArrayTransform(np.eye(4),"mm"))
    attrs: dict = field(default_factory=dict)
    date_modified: datetime = datetime.now()

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "session"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id
        if isinstance(self.targets, Point):
            self.targets = [self.targets]
        else:
            self.targets = list(self.targets)
        if isinstance(self.markers, Point):
            self.markers = [self.markers]
        else:
            self.markers = list(self.markers)

    @staticmethod
    def from_file(filename):
        """
        Create a Session from a file

        :param filename: Name of the file to read
        :param db: Database object
        :returns: Session object
        """
        with open(filename) as f:
            return Session.from_dict(json.load(f))

    @staticmethod
    def from_dict(d:Dict):
        """
        Create a session from a dictionary

        :param d: Dictionary of session parameters
        :param db: Database object
        :returns: Session object
        """
        if 'date' in d:
            d['date'] = datetime.fromisoformat(d['date'])
        if 'date_modified' in d:
            d['date_modified'] = datetime.fromisoformat(d['date_modified'])
        if 'volume' in d:
            raise ValueError("Sessions no longer recognize a volume attribute -- it is now volume_id.")
        if 'array_transform' in d:
            d['array_transform'] = ArrayTransform(np.array(d['array_transform']['matrix']), d['array_transform']['units'])
        if isinstance(d['targets'], list):
            if len(d['targets'])>0 and isinstance(d['targets'][0], dict):
                d['targets'] = [Point.from_dict(p) for p in d['targets']]
        elif isinstance(d['targets'], dict):
            d['targets'] = [Point.from_dict(d['targets'])]
        elif isinstance(d['targets'], Point):
            d['targets'] = [d['targets']]
        if isinstance(d['markers'], list):
            if len(d['markers'])>0 and isinstance(d['markers'][0], dict):
                d['markers'] = [Point.from_dict(p) for p in d['markers']]
        elif isinstance(d['markers'], dict):
            d['markers'] = [Point.from_dict(d['markers'])]
        elif isinstance(d['markers'], Point):
            d['markers'] = [d['markers']]
        return Session(**d)

    def to_dict(self):
        """
        Convert the session to a dictionary

        :returns: Dictionary of session parameters
        """
        d = self.__dict__.copy()
        d['date'] = d['date'].isoformat()
        d['date_modified'] = d['date_modified'].isoformat()
        d['targets'] = [p.to_dict() for p in d['targets']]
        d['markers'] = [p.to_dict() for p in d['markers']]

        d['array_transform'] = asdict(d['array_transform'])

        return d

    @staticmethod
    def from_json(json_string : str) -> "Session":
        """Load a Session from a json string"""
        return Session.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Session to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Session object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename):
        """
        Save the session to a file

        :param filename: Name of the file
        """
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))
