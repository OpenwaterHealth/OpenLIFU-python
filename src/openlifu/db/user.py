import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class User:
    id: str = "user"
    """ The unique identifier of the user """

    password_hash: Optional[str] = None
    """ A hashed user password for authentication. """

    roles: List[str] = field(default_factory=list)
    """ A list of roles """

    name: str = "User"
    """ The name of the user """

    description: str = ""
    """ A description of the user """

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def from_dict(d : Dict[str,Any]) -> "User":
        return User(**d)

    def to_dict(self):
        return {
            "id": self.id,
            "password_hash": self.password_hash,
            "roles": self.roles,
            "name": self.name,
            "description": self.description,
        }

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            d = json.load(f)
        return User.from_dict(d)

    @staticmethod
    def from_json(json_string : str) -> "User":
        """Load a User from a json string"""
        return User.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a User to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete User object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

    def to_file(self, filename: str):
        """
        Save the user to a file

        Args:
            filename: Name of the file
        """
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))
