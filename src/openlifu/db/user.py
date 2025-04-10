from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List

from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class User:
    id: Annotated[str, OpenLIFUFieldData("User ID", "The unique identifier of the user")] = "user"
    """The unique identifier of the user"""

    password_hash: Annotated[str, OpenLIFUFieldData("Password hash", "A hashed user password for authentication.")] = ""
    """A hashed user password for authentication."""

    roles: Annotated[List[str], OpenLIFUFieldData("Roles", "A list of roles")] = field(default_factory=list)
    """A list of roles"""

    name: Annotated[str, OpenLIFUFieldData("User name", "The name of the user")] = "User"
    """The name of the user"""

    description: Annotated[str, OpenLIFUFieldData("Description", "A description of the user")] = ""
    """A description of the user"""

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def from_dict(d : Dict[str,Any]) -> User:
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
    def from_json(json_string : str) -> User:
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
