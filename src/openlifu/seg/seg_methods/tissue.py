from dataclasses import dataclass
from openlifu.seg.seg_methods.seg_method import UniformSegmentation

@dataclass
class Tissue(UniformSegmentation):
    ref_material: str = "tissue"
