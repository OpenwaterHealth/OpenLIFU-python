from __future__ import annotations

import xarray as xa

from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod


class UniformSegmentation(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        return self._ref_segment(volume.coords)

class UniformTissue(UniformSegmentation):
    """ Assigns the tissue material to all voxels in the volume. """
    def __init__(self, materials: dict[str, Material] | None = None):
        if materials is None:
            materials = MATERIALS.copy()
        super().__init__(materials=materials, ref_material="tissue")

class UniformWater(UniformSegmentation):
    """ Assigns the water material to all voxels in the volume. """
    def __init__(self, materials: dict[str, Material] | None = None):
        if materials is None:
            materials = MATERIALS.copy()
        super().__init__(materials=materials, ref_material="water")
