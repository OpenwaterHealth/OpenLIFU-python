from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from openlifu.geo import Point
from openlifu.util.json import PYFUSEncoder
from openlifu.util.strings import sanitize


@dataclass
class ArrayTransform:
    """Class representing the transform on a transducer array to position it in space.
    units:
    """

    matrix: np.ndarray
    """4x4 affine transform matrix"""

    units : str
    """The units of the space on which to apply the transform matrix , e.g. "mm"
    (In order to apply the transform to transducer points,
    first represent the points in these units.)"""

@dataclass
class Session:
    """
    Class representing an openlifu session, which consists essentially of a patient scan, a protocol
    to use, potential targets for sonication, and a transducer situated in the patient space.
    """

    id: str | None = None
    """ID of this session"""

    subject_id: str | None = None
    """ID of the parent subject of this session"""

    name: str | None = None
    """Session name"""

    date_created: datetime = field(default_factory=datetime.now)
    """Date of creation of the session"""

    date_modified: datetime = field(default_factory=datetime.now)
    """Date of modification of the session"""

    protocol_id: str | None = None
    """ID of the protocol used for this session"""

    volume_id: str | None = None
    """ID of the subject volume associated with this session"""

    transducer_id: str | None = None
    """ID of the transducer associated with this session"""

    array_transform: ArrayTransform = field(default_factory=lambda : ArrayTransform(np.eye(4),"mm"))
    """The transducer affine transform matrix with units, situating the transducer in space"""

    targets: List[Point] = field(default_factory=list)
    """Targets saved to this session"""

    markers: List[Point] = field(default_factory=list)
    """Registration markers saved to this session"""

    attrs: dict = field(default_factory=dict)
    """Dictionary of additional custom attributes to save to the session"""

    virtual_fit_results: Dict[str,Tuple[bool,List[ArrayTransform]]] = field(default_factory=dict)
    """Virtual fit results. This is a dictionary mapping target IDs to pairs `(approval, transforms)`,
    where:

        `approval` is a boolean indicating whether the virtual fit for that target has been approved, and
        `transforms` is a list of transducer transforms resulting from the virtual fit for that target.

    The idea is that the list of transforms would be ordered from best to worst, and should of course
    contain at least one transform. The "approval" is intended to apply to the first transform in the list
    only. None of the other transforms in the list are considered to be approved.
    """

    transducer_tracking_approved: bool | None = False
    """Approval state of transducer tracking. `True` means the user has provided some kind of
    confirmation that the transducer transform in this session agrees with reality."""

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
        :returns: Session object
        """
        with open(filename) as f:
            return Session.from_dict(json.load(f))

    @staticmethod
    def from_dict(d:Dict):
        """
        Create a session from a dictionary

        :param d: Dictionary of session parameters
        :returns: Session object
        """
        if 'date_created' in d:
            d['date_created'] = datetime.fromisoformat(d['date_created'])
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
        if 'virtual_fit_results' in d:
            for target_id,(approval,transforms) in d['virtual_fit_results'].items():
                d['virtual_fit_results'][target_id] = (
                    approval,
                    [ArrayTransform(np.array(t_dict["matrix"]), t_dict["units"]) for t_dict in transforms],
                )
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
        d = copy.deepcopy(self.__dict__) # Deep copy needed so that we don't modify the internals of self below
        d['date_created'] = d['date_created'].isoformat()
        d['date_modified'] = d['date_modified'].isoformat()
        d['targets'] = [p.to_dict() for p in d['targets']]
        d['markers'] = [p.to_dict() for p in d['markers']]

        d['array_transform'] = asdict(d['array_transform'])

        for target_id,(approval,transforms) in d['virtual_fit_results'].items():
            d['virtual_fit_results'][target_id] = (
                approval,
                [asdict(t) for t in transforms],
            )

        return d

    @staticmethod
    def from_json(json_string : str) -> Session:
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
        Path(filename).parent.parent.mkdir(exist_ok=True) #sessions directory
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))

    def update_modified_time(self, time: datetime | None = None):
        if time is None:
            time = datetime.now()
        self.date_modified = time
