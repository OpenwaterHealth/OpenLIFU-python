from dataclasses import dataclass
from pyfus.seg.seg_methods.seg_method import SegmentationMethod
import xarray as xa

@dataclass
class SegmentMRI(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        raise NotImplementedError