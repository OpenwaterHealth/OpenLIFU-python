from dataclasses import dataclass
from pyfus.seg.seg_methods.seg_method import UniformSegmentation

@dataclass
class Water(UniformSegmentation):
    ref_material: str = "water"