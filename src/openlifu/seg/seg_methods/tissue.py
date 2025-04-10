from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from openlifu.seg.seg_methods.seg_method import UniformSegmentation
from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class Tissue(UniformSegmentation):
    ref_material: Annotated[str, OpenLIFUFieldData("Reference material", "Reference Material ID to use")] = "tissue"
    """Reference Material ID to use. For the Tissue dataclass, this should remain as 'tissue'."""
