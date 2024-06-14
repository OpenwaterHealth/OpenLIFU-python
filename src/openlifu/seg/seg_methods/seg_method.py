from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
from openlifu.seg.material import Material, MATERIALS
import xarray as xa
import numpy as np
from openlifu.seg import seg_methods

@dataclass
class SegmentationMethod:
    materials: dict = field(default_factory=lambda: MATERIALS.copy())
    ref_material: str = "water"
    
    def __post_init__(self):
        if self.ref_material not in self.materials.keys():
            raise ValueError(f"Reference material {self.ref_material} not found.")
    @abstractmethod
    def _segment(self, volume: xa.DataArray):
        pass

    @staticmethod
    def from_dict(d):
        if isinstance(d, str):
            import openlifu.seg.seg_methods        
            if d == "water":
                return openlifu.seg.seg_methods.Water()
            elif d == "tissue":
                return openlifu.seg.seg_methods.Tissue()
            elif d == "segmented":
                return openlifu.seg.seg_methods.SegmentMRI()
        else:
            d = d.copy()
            short_classname = d.pop("class")
            module_dict = seg_methods.__dict__
            class_constructor = module_dict[short_classname]
            return class_constructor(**d)

    def _material_indices(self, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        return {material_id: i for i, material_id in enumerate(materials.keys())}
    
    def _map_params(self, seg: xa.DataArray, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        material_dict = self._material_indices(materials=materials)
        params = xa.Dataset()
        ref_mat = materials[self.ref_material]
        for param_id in ref_mat.param_ids:
            info = Material.param_info(param_id)
            param = xa.DataArray(np.zeros(seg.shape), coords=seg.coords, attrs={"units": info["units"], "long_name": info["name"], "ref_value": ref_mat.get_param(param_id)})
            for material_id, material in materials.items():
                midx = material_dict[material_id]
                param.data[seg.data == midx] = getattr(material, param_id)
            params[param_id] = param
        params.attrs['ref_material'] = ref_mat
        return params
    
    def seg_params(self, volume: xa.DataArray, materials: Optional[dict] = None):
        materials = self.materials if materials is None else materials
        seg = self._segment(volume)
        params = self._map_params(seg, materials=materials)
        return params
    
    def ref_params(self, coords: xa.Coordinates):
       seg = self._ref_segment(coords)
       params = self._map_params(seg)
       return params
    
    def _ref_segment(self, coords: xa.Coordinates):
       material_dict = self._material_indices()
       m_idx = material_dict[self.ref_material]
       sz = list(coords.sizes.values())
       seg = xa.DataArray(np.full(sz, m_idx, dtype=int), coords=coords)
       return seg
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

@dataclass
class UniformSegmentation(SegmentationMethod):
    def _segment(self, vol: xa.DataArray):
       return self._ref_segment(vol.coords)