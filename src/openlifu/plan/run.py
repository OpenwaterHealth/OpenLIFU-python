from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.json import PYFUSEncoder


@dataclass
class Run:
    """
    Class representing a run
    """

    id: Annotated[str | None, OpenLIFUFieldData("Run ID", "ID of the run")] = None
    """ID of the run"""

    name: Annotated[str | None, OpenLIFUFieldData("Run name", "Name of the run")] = None
    """Name of the run"""

    success_flag: Annotated[bool | None, OpenLIFUFieldData("Success?", "True when run was successful, False otherwise")] = None
    """True when run was successful, False otherwise"""

    note: Annotated[str | None, OpenLIFUFieldData("Run notes", "Large text containing notes about the run")] = None
    """Large text containing notes about the run"""

    session_id: Annotated[str | None, OpenLIFUFieldData("Session ID", "Session ID")] = None
    """Session ID"""

    solution_id: Annotated[str | None, OpenLIFUFieldData("Solution ID", "Solution ID")] = None
    """Solution ID"""

    @staticmethod
    def from_file(filename):
        """
        Create a Run from a file

        :param filename: Name of the file to read
        :returns: Run object
        """
        with open(filename) as f:
            d = json.load(f)
        return Run.from_dict(d)

    @staticmethod
    def from_json(json_string : str) -> Run:
        """Load a Run from a json string"""
        return Run.from_dict(json.loads(json_string))

    @staticmethod
    def from_dict(d : Dict[str, Any]) -> Run:
        return Run(**d)

    def to_dict(self):
        """
        Convert the run to a dictionary

        :returns: Dictionary of run parameters
        """
        d = self.__dict__.copy()
        return d

    def to_json(self, compact:bool) -> str:
        """Serialize a Run to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Run object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename):
        """
        Save the Run to a file

        :param filename: Name of the file
        """
        Path(filename).parent.parent.mkdir(exist_ok=True)
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))
