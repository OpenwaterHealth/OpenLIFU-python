
import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu import Point


@pytest.fixture()
def example_point() -> Point:
    return Point(
        id = "example_point",
        name="Example point",
        color=(0.,0.7, 0.2),
        radius=1.5,
        position=np.array([-10.,0,25]),
        dims = ("R", "A", "S"),
        units = "m",
    )

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_point : Point, compact_representation: bool):
    reconstructed_point = example_point.from_json(example_point.to_json(compact_representation))
    assert dataclasses_are_equal(example_point, reconstructed_point)
