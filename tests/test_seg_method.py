from __future__ import annotations

import pytest

from openlifu.seg import MATERIALS, Material, SegmentationMethod, seg_methods
from openlifu.seg.seg_methods.uniform import UniformSegmentation


@pytest.fixture()
def example_seg_method() -> seg_methods.UniformWater:
    return seg_methods.UniformWater()

def test_seg_method_dict_conversion(example_seg_method : seg_methods.UniformSegmentation):
    assert SegmentationMethod.from_dict(example_seg_method.to_dict()) == example_seg_method

def test_seg_method_no_instantiate_abstract_class():
    with pytest.raises(TypeError):
        SegmentationMethod()  # pyright: ignore[reportAbstractUsage]

def test_uniform_seg_method_no_reference_material():
    with pytest.raises(ValueError, match="Reference material non_existent_material not found."):
        seg_method = seg_methods.UniformWater()
        seg_method.ref_material = 'non_existent_material'
        seg_method.__post_init__()

def test_uniformwater_errors_when_specify_ref_material():
    with pytest.raises(TypeError):
        seg_methods.UniformWater(
                ref_material = 'water',  # pyright: ignore[reportCallIssue]
            )

def test_materials_as_none_gets_default_materials():
    seg_method = seg_methods.UniformWater(materials=None)  # pyright: ignore[reportArgumentType]
    assert seg_method.materials == MATERIALS.copy()
