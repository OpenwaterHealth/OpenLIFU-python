from openlifu.db import Session, Database
from pathlib import Path
import pytest

@pytest.fixture()
def example_database() -> Database:
    return Database(Path(__file__).parent/'resources/example_db')

def test_load_session_from_file(example_database : Database):
    session = Session.from_file(
        filename = Path(__file__).parent/'resources/example_db/subjects/example_subject/sessions/example_session/example_session.json',
        db = example_database,
    )
    assert session.name == "Example Session"
    assert session.volume_id == "example_volume"
