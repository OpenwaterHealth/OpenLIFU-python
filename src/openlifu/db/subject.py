import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openlifu.io.dict_conversion import DictMixin
from openlifu.util.strings import sanitize


@dataclass
class Subject(DictMixin):
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
    def from_file(filename):
        """
        Create a subject from a file

        :param filename: Name of the file to read
        :returns: Subject object
        """
        with open(filename) as f:
            return Subject.from_dict(json.load(f))

    def to_file(self, filename):
        """
        Write the subject to a file

        :param filename: Name of the file to write
        """
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
