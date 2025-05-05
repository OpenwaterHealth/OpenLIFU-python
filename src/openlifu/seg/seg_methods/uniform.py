from __future__ import annotations

import xarray as xa

from openlifu.seg.seg_method import SegmentationMethod


class UniformSegmentation(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        return self._ref_segment(volume.coords)

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
