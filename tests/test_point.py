from dataclasses import fields

import numpy as np
import pytest

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
    for field in fields(example_point):
        value_original = getattr(example_point, field.name)
        value_reconstructed = getattr(reconstructed_point, field.name)
        if isinstance(value_original, np.ndarray):
            assert (value_original == value_reconstructed).all()
        else:
            assert value_original == value_reconstructed
