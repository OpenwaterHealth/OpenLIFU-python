import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import xarray

from openlifu.bf import Pulse, Sequence
from openlifu.geo import Point
from openlifu.util.json import PYFUSEncoder


@dataclass
class Solution:
    """
    A sonication solution resulting from beamforming and running a simulation.
    """
    id: str = "solution" # the *solution* id, a concept that did not exist in the matlab software
    """ID of this solution"""

    name: str = "Solution"
    """Name of this solution"""

    protocol_id: Optional[str] = None # this used to be called plan_id in the matlab code
    """ID of the protocol that was used when generating this solution"""

    transducer_id: Optional[str] = None
    """ID of the transducer that was used when generating this solution"""

    created_on: datetime = field(default_factory=datetime.now)
    """Solution creation time"""

    description: str = ""
    """Description of this solution"""

    delays: Optional[np.ndarray] = None
    """Vector of time delays to steer the beam"""

    apodizations: Optional[np.ndarray] = None
    """Vector of apodizations to steer the beam"""
    pulse: Pulse = field(default_factory=Pulse)
    """Pulse to send to the transducer when running sonication"""
    sequence: Sequence = field(default_factory=Sequence)
    """Pulse sequence to use when running sonication"""
    focus: Optional[Point] = None
    """Point that is being focused on in this Solution; part of the focal pattern of the target"""

    # there was "target_id" in the matlab software, but here we do not have the concept of a target ID.
    # I believe this was only needed in the matlab software because solutions were organized by target rather
    # than having their own unique solution ID. We do have unique solution IDs so it's possible we don't need
    # this target attribute at all here. Keeping it here for now just in case.
    target: Optional[Point] = None
    """The ultimate target of this sonication. This sonication solution is focused on one focal point
    in a pattern that is centered on this target."""

    # In the matlab code the simulation result was saved as a separate .mat file.
    # Here we include it as an xarray dataset.
    simulation_result: xarray.Dataset = field(default_factory=xarray.Dataset)
    """The xarray Dataset of simulation results"""

    def to_json(self, include_simulation_data:bool, compact:bool) -> str:
        """Serialize a Solution to a json string

        Args:
            include_array_data: if enabled then large simulation data arrays are serialized somehow into the json,
                so that they can be recovered via `from_json` alone. otherwise they are excluded.
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Solution object.
        """
        solution_dict = asdict(self)

        if include_simulation_data:
            # Serialize xarray dataset into a string
            solution_dict['simulation_result'] = base64.b64encode(self.simulation_result.to_netcdf()).decode('utf-8')
        else:
            solution_dict.pop('simulation_result')

        if compact:
            return json.dumps(solution_dict, separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(solution_dict, indent=4, cls=PYFUSEncoder)

    @staticmethod
    def from_json(json_string : str, simulation_result: Optional[xarray.Dataset]=None) -> "Solution":
        """Load a Solution from a json string.

        Args:
            json_string: the json string defining the Solution
            simulation_result: the simulation result arrays to use. If the json string has this then it will
                be read from the json string and it should not be provided in this argument.

        Returns: The new Solution object.
        """
        solution_dict = json.loads(json_string)
        solution_dict["created_on"] = datetime.fromisoformat(solution_dict["created_on"])
        if solution_dict["delays"] is not None:
            solution_dict["delays"] = np.array(solution_dict["delays"])
        if solution_dict["apodizations"] is not None:
            solution_dict["apodizations"] = np.array(solution_dict["apodizations"])
        solution_dict["pulse"] = Pulse.from_dict(solution_dict["pulse"])
        solution_dict["sequence"] = Sequence.from_dict(solution_dict["sequence"])
        if solution_dict["focus"] is not None:
            solution_dict["focus"] = Point.from_dict(solution_dict["focus"])
        if solution_dict["target"] is not None:
            solution_dict["target"] = Point.from_dict(solution_dict["target"])

        if simulation_result is not None:
            if "simulation_result" in solution_dict:
                raise ValueError(
                    "A simulation result was provided while the json string already contains `simulation_result`. "
                    "Unclear which to use!"
                )
            solution_dict["simulation_result"] = simulation_result
        elif "simulation_result" in solution_dict:
            # Deserialize xarray dataset from string
            solution_dict["simulation_result"] = xarray.open_dataset(base64.b64decode(
                solution_dict["simulation_result"].encode('utf-8')
            ))

        return Solution(**solution_dict)
