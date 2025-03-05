import logging
from pathlib import Path

import pytest

from openlifu.db import Session
from openlifu.vf import VirtualFit


@pytest.fixture()
def example_session() -> Session:
    return Session.from_file(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/example_session.json")

@pytest.mark.skip("wip")
def test_virtual_fit(
        example_session: Session
    ):
    """Test if virtual fit runs."""
    logging.disable(logging.CRITICAL)

    target = example_session.targets[0]
    vf = VirtualFit()
    vf.run(target)
