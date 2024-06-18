from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from pyfus.geo import Point
from pyfus.xdc import Transducer
from pyfus.util.strings import sanitize
import xarray

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
    volume: volume
    transducer: transducer
    attrs: Dictionary of attributes
    date_modified: Date of last modification
    """
    id: Optional[str] = None
    subject_id: Optional[str] = None
    name: Optional[str] = None
    date: datetime = datetime.now()
    targets: List[Point] = field(default_factory=list)
    markers: List[Point] = field(default_factory=list)
    volume: xarray.DataArray = field(default_factory=xarray.DataArray)
    transducer: Transducer = field(default_factory=Transducer)
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
    def from_dict(d):
        """
        Create a session from a dictionary
        
        :param d: Dictionary of session parameters
        :returns: Session object
        """
        if 'date' in d:
            d['date'] = datetime.fromisoformat(d['date'])
        if 'date_modified' in d:
            d['date_modified'] = datetime.fromisoformat(d['date_modified'])
        if isinstance(d['volume'], dict):
            d['volume'] = xarray.DataArray.from_dict(d['volume'])
        if isinstance(d['transducer'], dict):
            d['transducer'] = Transducer.from_dict(d['transducer'])
        if isinstance(d['targets'], list):
            if len(d['targets'])>0 and isinstance(d['targets'][0], dict):
                d['targets'] = [Point.from_dict(p) for p in d['targets']]
        else:
            if isinstance(d['targets'], dict):
                d['targets'] = [Point.from_dict(d['targets'])]
            elif isinstance(d['targets'], Point):
                d['targets'] = [d['targets']]
        if isinstance(d['markers'], list):
            if len(d['markers'])>0 and isinstance(d['markers'][0], dict):
                d['markers'] = [Point.from_dict(p) for p in d['markers']]
        else:
            if isinstance(d['markers'], dict):
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
        d['volume'] = d['volume'].to_dict()
        d['transducer'] = d['transducer'].to_dict()
        d['targets'] = [p.to_dict() for p in d['targets']]
        d['markers'] = [p.to_dict() for p in d['markers']]
        return d
    
    def to_file(self, filename):
        """
        Save the session to a file
        
        :param filename: Name of the file
        """
        from pyfus.util.json import to_json
        to_json(self.to_dict(), filename)