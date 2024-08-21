from openlifu import Transducer
from pathlib import Path
from dataclasses import fields
import numpy as np
import pytest

@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/'resources/example_db/transducers/example_transducer/example_transducer.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_transducer(example_transducer : Transducer, compact_representation: bool):
    reconstructed_transducer = example_transducer.from_json(example_transducer.to_json(compact_representation))
    for field in fields(example_transducer):
        value_original = getattr(example_transducer, field.name)
        value_reconstructed = getattr(reconstructed_transducer, field.name)
        if isinstance(value_original, np.ndarray):
            assert (value_original == value_reconstructed).all()
        else:
            assert value_original == value_reconstructed
