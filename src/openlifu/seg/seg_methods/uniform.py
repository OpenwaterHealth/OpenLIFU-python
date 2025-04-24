from __future__ import annotations

import xarray as xa

from openlifu.seg import SegmentationMethod
from openlifu.seg.material import MATERIALS, Material


class UniformSegmentation(SegmentationMethod):
    def __init__(self, ref_material: str = "water", materials: dict[str, Material] | None = None):
        self.ref_material = ref_material
        self.materials = materials if materials is not None else MATERIALS.copy()

    def _segment(self, vol: xa.DataArray):
        return self._ref_segment(vol.coords)

class UniformTissue(UniformSegmentation):
    """ Assigns the tissue material to all voxels in the volume. """
    def __init__(self, materials=None):
        super().__init__(materials=materials, ref_material="tissue")

    def to_dict(self):
        d = super().to_dict()
        d.pop("ref_material")
        d["class"] = "UniformTissue"
        return d

class UniformWater(UniformSegmentation):
    """ Assigns the water material to all voxels in the volume. """
    def __init__(self, materials=None):
        super().__init__(materials=materials, ref_material="water")

    def to_dict(self):
        d = super().to_dict()
        d.pop("ref_material")
        d["class"] = "UniformWater"
        return d
