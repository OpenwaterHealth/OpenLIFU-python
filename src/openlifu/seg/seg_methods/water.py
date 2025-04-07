from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from openlifu.seg.seg_methods.seg_method import UniformSegmentation
from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class Water(UniformSegmentation):
    ref_material: Annotated[str, OpenLIFUFieldData("Reference material", "Reference Material ID to use")] = "water"
    """Reference Material ID to use. For the Water dataclass, this should remain as 'water'."""
