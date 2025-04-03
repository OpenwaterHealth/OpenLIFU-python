from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.strings import sanitize


@dataclass
class Subject(DictMixin):
    """
    Class representing a subject
    """

    id: Annotated[str | None, OpenLIFUFieldData("Subject ID", "ID of the subject")] = None
    """ID of the subject"""

    name: Annotated[str | None, OpenLIFUFieldData("Subject name", "Name of the subject")] = None
    """Name of the subject"""

    attrs: Annotated[dict, OpenLIFUFieldData("Attributes", "Dictionary of attributes")] = field(default_factory=dict)
    """Dictionary of attributes"""

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "subject"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id

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
