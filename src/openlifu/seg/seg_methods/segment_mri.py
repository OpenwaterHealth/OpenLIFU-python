from dataclasses import dataclass

import xarray as xa

from openlifu.seg.seg_methods.seg_method import SegmentationMethod


@dataclass
class SegmentMRI(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        raise NotImplementedError
