import logging
import shutil
from pathlib import Path

import pytest
from helpers import dataclasses_are_equal

from openlifu import Solution
from openlifu.db import Database, Session, Subject


@pytest.fixture()
def example_database(tmp_path:Path) -> Database:
    """Example database in a temporary directory; appropriate to use when testing write operations."""
    shutil.copytree(Path(__file__).parent/'resources/example_db', tmp_path/"example_db")
    return Database(tmp_path/"example_db")

@pytest.fixture()
def example_session(example_database : Database) -> Session:
    return Session.from_file(
        filename = Path(example_database.path)/"subjects/example_subject/sessions/example_session/example_session.json",
    )

@pytest.fixture()
def example_subject(example_database : Database) -> Subject:
    return Subject.from_file(
        filename = Path(example_database.path)/"subjects/example_subject/example_subject.json",
    )

def test_load_session_from_file(example_session : Session, example_database : Database):

    # Test that Session loaded via Session.from_file is correct
    session = example_session
    assert session.name == "Example Session"
    assert session.volume_id == "example_volume"
    assert session.transducer_id == "example_transducer"
    assert session.protocol_id == "example_protocol"
    assert session.array_transform.matrix.shape == (4,4)
    assert session.array_transform.units == "mm"

    # Test that the Session loaded via the Database is identical
    session_from_database = example_database.load_session(
        example_database.load_subject(session.subject_id),
        session.id,
    )
    assert dataclasses_are_equal(session_from_database,session)

def test_write_subject(example_database : Database):
    subject = Subject(id="bleh",name="Seb Jectson")

    # Can add a new subject, and it loads back in correctly.
    example_database.write_subject(subject)
    reloaded_subject = example_database.load_subject("bleh")
    assert subject == reloaded_subject

    # Error raised when the subject already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_subject(subject, on_conflict="error")

    # Skip option
    subject.name = "Deb Jectson"
    example_database.write_subject(subject, on_conflict="skip")
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Seb Jectson"

    # Overwrite option
    example_database.write_subject(subject, on_conflict="overwrite")
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Deb Jectson"

def test_write_session(example_database: Database, example_subject: Subject):
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)

    # Can add a new session, and it loads back in correctly.
    example_database.write_session(example_subject, session)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert dataclasses_are_equal(reloaded_session,session)

    # Error raised when the session already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_session(example_subject, session, on_conflict="error")

    # Skip option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict="skip")
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "bleh"

    # Overwrite option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict="overwrite")
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "new_name"

def test_write_session_mismatched_id(example_database: Database, example_subject: Subject):
    session = Session(id='a_session',subject_id='bogus_id') # The subject ID here is different from the ID in example_subject
    with pytest.raises(ValueError, match="IDs do not match"):
        example_database.write_session(example_subject, session)

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_session(example_session : Session, compact_representation:bool):
    reconstructed_session = example_session.from_json(example_session.to_json(compact_representation))
    assert dataclasses_are_equal(example_session, reconstructed_session)

def test_session_to_file(example_session : Session, tmp_path:Path):
    save_path = tmp_path/"this_is_a_session.json"
    example_session.to_file(save_path)
    reloaded_session = Session.from_file(save_path)
    assert dataclasses_are_equal(example_session, reloaded_session)

def test_get_solutions_filename(example_database:Database):
    solutions_filepath = example_database.get_solutions_filename("example_subject", "example_session")
    assert solutions_filepath.exists()
    assert solutions_filepath.is_file()
    assert solutions_filepath.name == "solutions.json"

def test_get_solution_filepath(example_database:Database):
    solutions_dir = example_database.get_solution_filepath("example_subject", "example_session", "example_solution")
    assert solutions_dir.exists()
    assert solutions_dir.is_file()
    assert solutions_dir.name == "example_solution.json"

def test_get_solution_ids(example_database:Database, caplog):
    # verify that solution ids are loaded correctly
    solution_ids = example_database.get_solution_ids("example_subject", "example_session")
    assert len(solution_ids) == 1
    assert solution_ids[0] == "example_solution"

    # verify that warning is printed and empty list returned when there is no solutions file
    solutions_filepath = example_database.get_solutions_filename("example_subject", "example_session")
    solutions_filepath.unlink() # Delete file
    with caplog.at_level(logging.WARNING):
        solution_ids = example_database.get_solution_ids("example_subject", "example_session")
        assert "Solutions file not found" in caplog.text
    assert len(solution_ids) == 0


def test_load_solution(example_database:Database, example_session:Session):
    with pytest.raises(FileNotFoundError,match="Solution file not found"):
        example_database.load_solution(example_session, "bogus_solution_id")

    example_solution = example_database.load_solution(example_session, "example_solution")
    assert example_solution.name == "Example Solution"
    assert "p_min" in example_solution.simulation_result.data_vars # ensure the xarray dataset got loaded too

def test_write_solution(example_database:Database, example_session:Session):
    solution = Solution(name="bleh", id='new_solution')

    # This solution is not initially in the list of solution IDs
    assert solution.id not in example_database.get_solution_ids(example_session.subject_id, example_session.id)

    # Can add a new solution, and it loads back in correctly.
    example_database.write_solution(example_session, solution)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert dataclasses_are_equal(reloaded_solution,solution)

    # The new solution has now been added to the list of solution IDs
    assert solution.id in example_database.get_solution_ids(example_session.subject_id, example_session.id)

    # Error raised when the solution already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_solution(example_session, solution, on_conflict="error")

    # Skip option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict="skip")
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "bleh"

    # Overwrite option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict="overwrite")
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "new_name"
