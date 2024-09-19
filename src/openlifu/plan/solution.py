from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import xarray

from openlifu.bf import Pulse, Sequence
from openlifu.geo import Point


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
