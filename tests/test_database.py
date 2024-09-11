from openlifu.db import Session, Database, Subject
from pathlib import Path
import pytest
import shutil

@pytest.fixture()
def example_database(tmp_path:Path) -> Database:
    """Example database in a temporary directory; appropriate to use when testing write operations."""
    shutil.copytree(Path(__file__).parent/'resources/example_db', tmp_path/"example_db")
    return Database(tmp_path/"example_db")



def test_load_session_from_file(example_database : Database):
    session = Session.from_file(
        filename = Path(example_database.path)/"subjects/example_subject/sessions/example_session/example_session.json",
        db = example_database,
    )
    assert session.name == "Example Session"
    assert session.volume_id == "example_volume"

def test_add_subject(example_database : Database):
    subject = Subject(id="bleh",name="Seb Jectson")

    # Can add a new subject, and it loads back in correctly.
    example_database.add_subject(subject)
    reloaded_subject = example_database.load_subject("bleh")
    assert subject == reloaded_subject

    # Error raised when the subject already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.add_subject(subject, on_conflict="error")

    # Skip option
    subject.name = "Deb Jectson"
    example_database.add_subject(subject, on_conflict="skip")
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Seb Jectson"

    # Overwrite option
    example_database.add_subject(subject, on_conflict="overwrite")
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Deb Jectson"
