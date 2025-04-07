from __future__ import annotations

import pytest

from openlifu.seg import Material, SegmentationMethod


@pytest.fixture()
def example_seg_method() -> SegmentationMethod:
    return SegmentationMethod(
        materials = {
            'water' : Material(
                id="water",
                name="water",
                sound_speed=1500.0,
                density=1000.0,
                attenuation=0.0,
                specific_heat=4182.0,
                thermal_conductivity=0.598
            ),
            'skull' : Material(
                id="skull",
                name="skull",
                sound_speed=4080.0,
                density=1900.0,
                attenuation=0.0,
                specific_heat=1100.0,
                thermal_conductivity=0.3
            ),
        },
        ref_material = 'water',
    )

def test_seg_method_dict_conversion(example_seg_method : SegmentationMethod):
    assert SegmentationMethod.from_dict(example_seg_method.to_dict()) == example_seg_method
