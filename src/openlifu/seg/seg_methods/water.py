from dataclasses import dataclass

from openlifu.seg.seg_methods.seg_method import UniformSegmentation


@dataclass
class Water(UniformSegmentation):
    ref_material: str = "water"
