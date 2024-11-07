import logging
from pathlib import Path

import pytest

from openlifu import Protocol, Transducer
from openlifu.bf.focal_patterns import Wheel
from openlifu.db import Session
from openlifu.plan.protocol import OnPulseMismatchAction
from openlifu.plan.target_constraints import TargetConstraints


@pytest.fixture()
def example_protocol() -> Protocol:
    return Protocol.from_file(Path(__file__).parent/'resources/example_db/protocols/example_protocol/example_protocol.json')

@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/"resources/example_db/transducers/example_transducer/example_transducer.json")

@pytest.fixture()
def example_session() -> Session:
    return Session.from_file(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/example_session.json")

@pytest.fixture()
def example_wheel_pattern() -> Wheel:
    return Wheel(num_spokes=6)

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_protocol : Protocol, compact_representation: bool):
    assert example_protocol.from_json(example_protocol.to_json(compact_representation)) == example_protocol

def test_default_protocol():
    """Ensure it is possible to construct a default protocol"""
    Protocol()

@pytest.mark.parametrize(
    "target_constraints",
    [
        [
            TargetConstraints(dim="P", units="mm", min=0.0, max=float("inf")),
        ],
        [
            TargetConstraints(dim="P", units="m", min=-0.001, max=0.0),
        ],
        [
            TargetConstraints(dim="L", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="P", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="S", units="mm", min=-100.0, max=-10.0),
        ]
    ]
)
def test_check_target(example_protocol: Protocol, example_session: Session, target_constraints: TargetConstraints):
    """Ensure that the target can be correctly verified."""
    example_protocol.target_constraints = target_constraints
    with pytest.raises(ValueError, match="not within bounds"):
        example_protocol.check_target(example_session.targets[0])

@pytest.mark.parametrize("on_pulse_mismatch", [
            OnPulseMismatchAction.ERROR,
            OnPulseMismatchAction.ROUND,
            OnPulseMismatchAction.ROUNDUP,
            OnPulseMismatchAction.ROUNDDOWN
        ]
    )
def test_fix_pulse_mismatch(
        example_protocol: Protocol,
        example_session: Session,
        example_wheel_pattern: Wheel,
        on_pulse_mismatch: OnPulseMismatchAction
    ):
    """Test if sequence is correctly fixed for all pulse mismatch actions."""
    logging.disable(logging.CRITICAL)

    target = example_session.targets[0]
    foci = example_wheel_pattern.get_targets(target)
    num_foci = len(foci)
    if on_pulse_mismatch is OnPulseMismatchAction.ERROR:
        with pytest.raises(ValueError, match="not a multiple of the number of foci"):
            example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
    else:
        example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
        if on_pulse_mismatch is OnPulseMismatchAction.ROUND:
            assert example_protocol.sequence.pulse_count == num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDUP:
            assert example_protocol.sequence.pulse_count == 2*num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDDOWN:
            assert example_protocol.sequence.pulse_count == num_foci
