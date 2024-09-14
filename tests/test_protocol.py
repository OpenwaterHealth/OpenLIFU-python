from pathlib import Path

import pytest

from openlifu import Protocol


@pytest.fixture()
def example_protocol() -> Protocol:
    return Protocol.from_file(Path(__file__).parent/'resources/example_db/protocols/example_protocol/example_protocol.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_protocol : Protocol, compact_representation: bool):
    assert example_protocol.from_json(example_protocol.to_json(compact_representation)) == example_protocol

def test_default_protocol():
    """Ensure it is possible to construct a default protocol"""
    Protocol()
