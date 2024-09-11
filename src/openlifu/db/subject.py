from dataclasses import dataclass, field
from typing import List, Optional
from openlifu.util.strings import sanitize
import json
from pathlib import Path

@dataclass
class Subject:
    """
    Class representing a subject

    ivar id: ID of the subject
    ivar name: Name of the subject
    ivar volumes: List of volume IDs
    ivar attrs: Dictionary of attributes
    """
    id: Optional[str] = None
    name: Optional[str] = None
    volumes: List[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "subject"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id
        if isinstance(self.volumes, str):
            self.volumes = [self.volumes]
        else:
            self.volumes = list(self.volumes)

    @staticmethod
    def from_dict(d):
        """
        Create a subject from a dictionary

        :param d: Dictionary of subject parameters
        :returns: Subject object
        """
        return Subject(**d)

    @staticmethod
    def from_file(filename):
        """
        Create a subject from a file

        :param filename: Name of the file to read
        :returns: Subject object
        """
        with open(filename, 'r') as f:
            return Subject.from_dict(json.load(f))

    def to_dict(self):
        """
        Convert the subject to a dictionary

        :returns: Dictionary of subject parameters
        """
        return self.__dict__.copy()

    def to_file(self, filename):
        """
        Write the subject to a file

        :param filename: Name of the file to write
        """
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
