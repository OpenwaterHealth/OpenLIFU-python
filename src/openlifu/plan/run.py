import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openlifu.util.json import PYFUSEncoder


@dataclass
class Run:
    """
    Class representing a run
    id: ID of the run
    success_flag: True when run was successful, False otherwise
    note: large text containing notes about the run
    session_id: session id
    solution_id: solution id
    """
    id: Optional[str] = None
    success_flag: Optional[bool] = None
    note: Optional[str] = None
    session_id: Optional[str] = None
    solution_id: Optional[str] = None

    @staticmethod
    def from_file(filename):
        """
        Create a Run from a file

        :param filename: Name of the file to read
        :returns: Run object
        """
        with open(filename) as f:
            return Run.from_dict(json.load(f))

    @staticmethod
    def from_json(json_string : str) -> "Run":
        """Load a Run from a json string"""
        return Run.from_dict(json.loads(json_string))

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
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))
