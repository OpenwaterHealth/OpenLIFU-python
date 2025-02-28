from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu import Transducer


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/'resources/example_db/transducers/example_transducer/example_transducer.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_transducer(example_transducer : Transducer, compact_representation: bool):
    reconstructed_transducer = example_transducer.from_json(example_transducer.to_json(compact_representation))
    dataclasses_are_equal(example_transducer, reconstructed_transducer)

def test_get_polydata_color_options(example_transducer : Transducer):
    """Ensure that the color is set correctly on the polydata"""
    polydata_with_default_color = example_transducer.get_polydata()
    point_scalars = polydata_with_default_color.GetPointData().GetScalars()
    assert point_scalars is None

    polydata_with_given_color = example_transducer.get_polydata(facecolor=[0,1,1,0.5])
    point_scalars = polydata_with_given_color.GetPointData().GetScalars()
    assert point_scalars is not None

def test_default_transducer():
    """Ensure it is possible to construct a default transducer"""
    Transducer()

def test_convert_transform():
    transducer = Transducer(units='cm')
    transform = transducer.convert_transform(
        matrix = np.array([
            [1,0,0,2],
            [0,1,0,3],
            [0,0,1,4],
            [0,0,0,1],
        ], dtype=float),
        units = "m",
    )
    expected_transform = np.array([
        [1,0,0,200],
        [0,1,0,300],
        [0,0,1,400],
        [0,0,0,1],
    ], dtype=float)
    assert np.allclose(transform,expected_transform)

def test_get_effective_origin():
    transducer = Transducer.gen_matrix_array(nx=3, ny=2, units='cm')
    effective_origin_with_all_active = transducer.get_effective_origin(apodizations = np.ones(transducer.numelements()))
    assert np.allclose(effective_origin_with_all_active, np.zeros(3))

    rng = np.random.default_rng()
    element_index_to_turn_on = rng.integers(transducer.numelements())
    apodizations_with_just_one_element = np.zeros(transducer.numelements())
    apodizations_with_just_one_element[element_index_to_turn_on] = 0.5 # It is allowed to be a number between 0 and 1
    assert np.allclose(
        transducer.get_effective_origin(apodizations = apodizations_with_just_one_element, units = "um"),
        transducer.get_positions(units="um")[element_index_to_turn_on],
    )
