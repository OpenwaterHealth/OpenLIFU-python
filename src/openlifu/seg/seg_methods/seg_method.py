from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np
import xarray as xa

from openlifu.seg import seg_methods
from openlifu.seg.material import MATERIALS, Material


@dataclass
class SegmentationMethod:
    materials: dict[str,Material] = field(default_factory=lambda: MATERIALS.copy())
    ref_material: str = "water"

    def __post_init__(self):
        if self.ref_material not in self.materials:
            raise ValueError(f"Reference material {self.ref_material} not found.")
    @abstractmethod
    def _segment(self, volume: xa.DataArray):
        pass

    @staticmethod
    def from_dict(d):
        if isinstance(d, str):
            if d == "water":
                return seg_methods.Water()
            elif d == "tissue":
                return seg_methods.Tissue()
            elif d == "segmented":
                return seg_methods.SegmentMRI()
        else:
            d = copy.deepcopy(d)
            short_classname = d.pop("class")
            if "materials" in d:
                for material_key, material_definition in d["materials"].items():
                    if not isinstance(material_definition, Material): # if it is given as a dict rather than a fully hydrated object
                        param_ids = material_definition.pop("param_ids") # param_ids is a field of the dataclass, but it has init=False
                        d["materials"][material_key] = Material.from_dict(material_definition)
                        d["materials"][material_key].param_ids = tuple(param_ids)
            module_dict = seg_methods.__dict__
            class_constructor = module_dict[short_classname]
            return class_constructor(**d)

    def _material_indices(self, materials: dict | None = None):
        materials = self.materials if materials is None else materials
        return {material_id: i for i, material_id in enumerate(materials.keys())}

    def _map_params(self, seg: xa.DataArray, materials: dict | None = None):
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

    def seg_params(self, volume: xa.DataArray, materials: dict | None = None):
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
        d = asdict(self)
        d['class'] = self.__class__.__name__
        return d

@dataclass
class UniformSegmentation(SegmentationMethod):
    def _segment(self, vol: xa.DataArray):
        return self._ref_segment(vol.coords)
