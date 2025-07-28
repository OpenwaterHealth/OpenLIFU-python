from __future__ import annotations

import pandas as pd
import xarray as xa

from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod


class UniformSegmentation(SegmentationMethod):
    def _segment(self, volume: xa.DataArray):
        return self._ref_segment(volume.coords)

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the segmentation method parameters

        :returns: Pandas DataFrame of the segmentation method parameters
        """
        records = [{"Name": "Type", "Value": "Uniform Tissue", "Unit": ""},
                   {"Name": "Reference Material", "Value": self._ref_material, "Unit": ""}]
        return pd.DataFrame.from_records(records)

class UniformTissue(UniformSegmentation):
    """ Assigns the tissue material to all voxels in the volume. """
    def __init__(self, materials: dict[str, Material] | None = None):
        if materials is None:
            materials = MATERIALS.copy()
        super().__init__(materials=materials, ref_material="tissue")

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the segmentation method parameters

        :returns: Pandas DataFrame of the segmentation method parameters
        """
        records = [{"Name": "Type", "Value": "Uniform Tissue", "Unit": ""}]
        return pd.DataFrame.from_records(records)

class UniformWater(UniformSegmentation):
    """ Assigns the water material to all voxels in the volume. """
    def __init__(self, materials: dict[str, Material] | None = None):
        if materials is None:
            materials = MATERIALS.copy()
        super().__init__(materials=materials, ref_material="water")

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the segmentation method parameters

        :returns: Pandas DataFrame of the segmentation method parameters
        """
        records = [{"Name": "Type", "Value": "Uniform Water", "Unit": ""}]
        return pd.DataFrame.from_records(records)
