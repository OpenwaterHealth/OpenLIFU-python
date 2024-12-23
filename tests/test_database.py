import logging
import shutil
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Optional

import pytest
from helpers import dataclasses_are_equal

from openlifu import Point, Solution
from openlifu.db import Session, Subject
from openlifu.db.database import Database, OnConflictOpts
from openlifu.plan import Protocol, Run


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

def test_write_protocol(example_database: Database):
    protocol = Protocol(name="bleh", id="a_protocol_called_bleh")

    # Protocol id is not in list initially
    assert protocol.id not in example_database.get_protocol_ids()

    # Can add a new protocol, and it loads back in correctly.
    example_database.write_protocol(protocol)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert dataclasses_are_equal(reloaded_protocol,protocol)

    # Protocol id is now in the list
    assert protocol.id in example_database.get_protocol_ids()

    # Error raised when the protocol already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_protocol(protocol, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    protocol.name = "new_name"
    example_database.write_protocol(protocol, on_conflict=OnConflictOpts.SKIP)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert reloaded_protocol.name == "bleh"

    # Overwrite option
    protocol.name = "new_name"
    example_database.write_protocol(protocol, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert reloaded_protocol.name == "new_name"

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

    # Empty sessions file is created (overlaps with test_write_subject_associated_object_structure_created but it's okay)
    sessions_filename = example_database.get_sessions_filename(subject.id)
    assert sessions_filename.exists()
    assert sessions_filename.is_file()
    assert sessions_filename.name == "sessions.json"
    session_ids = example_database.get_session_ids(subject.id)
    assert session_ids == []

    # Add a session so we can later test that overwriting a subject doesn't wipe out the session
    session = Session(id="jectson_session_1", subject_id=subject.id)
    example_database.write_session(subject, session)

    # Error raised when the subject already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_subject(subject, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    subject.name = "Deb Jectson"
    example_database.write_subject(subject, on_conflict=OnConflictOpts.SKIP)
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Seb Jectson"

    # Overwrite option
    example_database.write_subject(subject, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Deb Jectson"

    # Ensure that after overwrite of a subject the sessions are still there
    assert session.id in example_database.get_session_ids(subject.id)
    assert dataclasses_are_equal(
        example_database.load_session(subject, session.id),
        session
    )

def test_write_subject_associated_object_structure_created(example_database : Database):
    """Test that when you create a new subject, the file structure needed for other objects that can be written
    under that subject is also created."""
    subject = Subject(id="bleh",name="Seb Jectson")
    example_database.write_subject(subject)

    assert example_database.get_sessions_filename(subject.id).is_file()
    assert example_database.get_volumes_filename(subject.id).is_file()

def test_write_session(example_database: Database, example_subject: Subject):
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)

    # Can add a new session, and it loads back in correctly.
    example_database.write_session(example_subject, session)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert dataclasses_are_equal(reloaded_session,session)

    # Add a solution and a run to later test that overwriting a session doesn't wipe them out
    solution = Solution(id="please_keep_me")
    run = Run(id="please_keep_me_too")
    example_database.write_solution(session,solution)
    example_database.write_run(run,session)

    # Error raised when the session already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.SKIP)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "bleh"

    # Overwrite option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "new_name"

    # Ensure that after overwrite of a session the runs and solutions are still there
    assert solution.id in example_database.get_solution_ids(session.subject_id, session.id)
    assert dataclasses_are_equal(
        example_database.load_solution(session, solution.id),
        solution,
    )
    assert run.id in example_database.get_run_ids(session.subject_id, session.id)

    # When writing to a new subject
    new_subject = Subject(id="bleh_new",name="Deb Jectson")
    example_database.write_subject(new_subject, on_conflict=OnConflictOpts.OVERWRITE)
    session = Session(name="bleh", id='a_session',subject_id=new_subject.id)
    example_database.write_session(new_subject, session)
    reloaded_session = example_database.load_session(new_subject, session.id)
    assert reloaded_session.name == "bleh"

def test_write_session_associated_object_structure_created(example_database: Database, example_subject: Subject):
    """Test that when you create a new session, the file structure needed for other objects that can be written
    under that session is also created."""
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)
    example_database.write_session(example_subject, session)

    assert example_database.get_solutions_filename(example_subject.id, session.id).is_file()
    assert example_database.get_runs_filename(example_subject.id, session.id).is_file()


def test_write_run(example_database: Database, tmp_path:Path):
    subject_id = "example_subject"
    session_id = "example_session"
    protocol_id = "example_protocol"
    run_id = "example_run_2"
    success_flag = True
    note = "Test note"
    solution_id = "example_solution"
    subject = example_database.load_subject(subject_id)
    session = example_database.load_session(subject, session_id)
    protocol = example_database.load_protocol(protocol_id)
    run = Run(id=run_id, success_flag=success_flag, note=note, session_id=session_id, solution_id=solution_id)

    # Can add a new run
    example_database.write_run(run, session, protocol)
    run_file_path = tmp_path/"example_db/subjects/example_subject/sessions/example_session/runs/example_run/example_run.json"
    assert(run_file_path.is_file())

    # Error raised when the session already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_run(run, session, protocol, on_conflict=OnConflictOpts.ERROR)

    # Error raised when the user try to overwrite a run
    with pytest.raises(ValueError, match="may not be overwritten"):
        example_database.write_run(run, session, protocol, on_conflict=OnConflictOpts.OVERWRITE)

    # Make sure the Runs folder and Runs file are created
    new_session = Session(id='new_session',subject_id='example_subject')
    example_database.write_session(subject, new_session)
    new_run = Run(id=run_id, success_flag=success_flag, note=note, session_id='new_session', solution_id=solution_id)
    example_database.write_run(new_run, new_session, protocol, on_conflict=OnConflictOpts.OVERWRITE)
    runs_filename = example_database.get_runs_filename(subject.id, new_session.id)
    assert runs_filename.exists()
    assert runs_filename.is_file()
    assert runs_filename.name == "runs.json"

def test_load_session_snapshot(example_database: Database):
    subject_id = "example_subject"
    session_id = "example_session"
    run_id = "example_run"
    session = example_database.load_session_snapshot(subject_id, session_id, run_id)
    assert session.id == "example_session"

def test_load_protocol_snapshot(example_database: Database):
    subject_id = "example_subject"
    session_id = "example_session"
    run_id = "example_run"
    protocol = example_database.load_protocol_snapshot(subject_id, session_id, run_id)
    assert protocol.id == "example_protocol"

def test_write_session_mismatched_id(example_database: Database, example_subject: Subject):
    session = Session(id='a_session',subject_id='bogus_id') # The subject ID here is different from the ID in example_subject
    with pytest.raises(ValueError, match="IDs do not match"):
        example_database.write_session(example_subject, session)

@pytest.mark.parametrize(
    ("virtual_fit_approval_for_target_id", "expectation"),
    [
        (None, does_not_raise()), # see https://docs.pytest.org/en/6.2.x/example/parametrize.html#parametrizing-conditional-raising
        ("an_existing_target_id", does_not_raise()),
        ("bogus_target_id", pytest.raises(ValueError, match="virtual_fit_approval_for_target_id.*not in")),
    ]
)
def test_write_session_with_invalid_fit_approval(
    example_database: Database,
    example_subject: Subject,
    virtual_fit_approval_for_target_id: Optional[str],
    expectation,
):
    """Verify that writing a session with fit approval raises the invalid target error if and only if the
    target being approved does not exist."""
    session = Session(
        id="unique_id_2764592837465",
        subject_id=example_subject.id,
        targets=[Point(id="an_existing_target_id")],
        virtual_fit_approval_for_target_id=virtual_fit_approval_for_target_id,
    )
    with expectation:
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

def test_get_volume_info(example_database:Database, tmp_path:Path):
    subject = "example_subject"
    volume_id = "example_volume"
    volume_info = example_database.get_volume_info(subject, volume_id)
    assert(volume_info["id"] == "example_volume")
    assert(volume_info["name"] == "EXAMPLE_VOLUME")
    assert(Path(volume_info["data_abspath"]) == \
                        tmp_path/"example_db/subjects/example_subject/volumes/example_volume/example_volume.nii")

def test_get_volume_ids(example_database:Database):
    assert(example_database.get_volume_ids("example_subject") == ["example_volume"])

def test_write_volume_ids(example_database:Database):
    example_database.write_volume_ids("example_subject", ["example_volume", "example_volume_2"])
    assert(example_database.get_volume_ids("example_subject") == ["example_volume", "example_volume_2"])

def test_get_volume_dir(example_database:Database, tmp_path:Path):
    subject_id = "example_subject"
    volume_id = "example_volume"
    assert(example_database.get_volume_dir(subject_id, volume_id) == \
                        tmp_path/f'example_db/subjects/{subject_id}/volumes/{volume_id}')

def test_write_volume(example_database:Database, tmp_path:Path):
    subject_id = "example_subject"
    volume_id = "example_volume_2"
    volume_name = "EXAMPLE_VOLUME_2"
    volume_data_path = Path(tmp_path/'test_db_files/example_volume_2.nii')
    volume_data_path.parent.mkdir(parents=True, exist_ok=True)
    volume_data_path.touch()
    example_database.write_volume(subject_id, volume_id, volume_name, volume_data_path)
    assert(example_database.get_volume_ids("example_subject") == ["example_volume", "example_volume_2"])

    volume_filepath = example_database.get_volume_metadata_filepath("example_subject", "example_volume_2")
    assert(volume_filepath.name == "example_volume_2.json")
    assert((volume_filepath.parent/"example_volume_2.nii").exists())

    # When writing to a new subject
    subject = Subject(id="bleh",name="Deb Jectson")
    example_database.write_subject(subject, on_conflict=OnConflictOpts.OVERWRITE)
    example_database.write_volume(subject.id, volume_id, volume_name, volume_data_path)

    assert(example_database.get_volume_ids("bleh") == ["example_volume_2"])

    volume_filepath = example_database.get_volume_metadata_filepath("bleh", "example_volume_2")
    assert(volume_filepath.name == "example_volume_2.json")
    assert((volume_filepath.parent/"example_volume_2.nii").exists())

def test_load_solution(example_database:Database, example_session:Session):
    with pytest.raises(FileNotFoundError,match="Solution file not found"):
        example_database.load_solution(example_session, "bogus_solution_id")

    example_solution = example_database.load_solution(example_session, "example_solution")
    assert example_solution.name == "Example Solution"
    assert "p_min" in example_solution.simulation_result.data_vars # ensure the xarray dataset got loaded too

    # ensure the simulation and beamform data was loaded for all foci
    assert len(example_solution.simulation_result['focal_point_index']) == len(example_solution.foci)
    assert example_solution.delays.shape[0] == len(example_solution.foci)
    assert example_solution.apodizations.shape[0] == len(example_solution.foci)

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
        example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.SKIP)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "bleh"

    # Overwrite option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "new_name"

def test_write_solution_new_session(example_database:Database, example_subject:Subject):
    """Writing a solution should be possible in a newly created session"""
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)
    solution = Solution(name="bleh", id='new_solution')
    example_database.write_session(example_subject, session)
    example_database.write_solution(session, solution)

def test_get_photoscan_info(example_database:Database):
    subject_id = "example_subject"
    session_id = "example_session"
    photoscan_id = "example_photoscan"
    photoscan_info = example_database.get_photoscan_info(subject_id, session_id, photoscan_id)
    assert(photoscan_info["id"] == "example_photoscan")
    assert(photoscan_info["name"] == "ExamplePhotoscan")
    assert(Path(photoscan_info["model_abspath"]).exists())
    assert(Path(photoscan_info["texture_abspath"]).exists())

def test_get_photoscan_ids(example_database:Database):
    assert(example_database.get_photoscan_ids("example_subject", "example_session") == ["example_photoscan"])

def test_write_photoscan(example_database:Database, example_session: Session, tmp_path:Path):
    photoscan_id = "example_photoscan_2"
    photoscan_name = "EXAMPLE_PHOTOSCAN_2"
    model_data_path = Path(tmp_path/"test_db_files/example_photoscan_2.obj")
    model_data_path.parent.mkdir(parents=True, exist_ok=True)
    model_data_path.touch()
    texture_data_path = Path(tmp_path/"test_db_files/example_photoscan_texture_2.exr")
    texture_data_path.parent.mkdir(parents=True, exist_ok=True)
    texture_data_path.touch()
    mtl_data_path = Path(tmp_path/"test_db_files/example_photoscan.mtl")
    mtl_data_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_data_path.touch()
    example_database.write_photoscan(example_session, photoscan_id, photoscan_name, model_data_path, texture_data_path, mtl_data_path)
    assert(len(example_database.get_photoscan_ids("example_subject", "example_session")) == 2)
    assert("example_photoscan" in example_database.get_photoscan_ids("example_subject", "example_session"))
    assert("example_photoscan_2" in example_database.get_photoscan_ids("example_subject", "example_session"))

    photoscan_filepath = example_database.get_photoscan_metadata_filepath("example_subject","example_session","example_photoscan_2")
    assert(photoscan_filepath.name == "example_photoscan_2.json")
    assert((photoscan_filepath.parent/"example_photoscan_2.obj").exists())
    assert((photoscan_filepath.parent/"example_photoscan_texture_2.exr").exists())
    assert((photoscan_filepath.parent/"example_photoscan.mtl").exists())

    # Test not existent filepath
    bogus_texture_file = Path(tmp_path/"test_db_files/bogus_photoscan.exr")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        example_database.write_photoscan(example_session, photoscan_id, photoscan_name, model_data_path, bogus_texture_file, on_conflict=OnConflictOpts.OVERWRITE)

    # When writing to a new subject and session
    subject = Subject(id="bleh_photoscan_test",name="Deb Jectson")
    example_database.write_subject(subject)
    session = Session(id = "bleh_session", subject_id=subject.id, name = "Bleh_Session")
    example_database.write_session(subject, session)
    example_database.write_photoscan(session, photoscan_id, photoscan_name, model_data_path, texture_data_path)

    assert(example_database.get_photoscan_ids(subject.id,session.id) == ["example_photoscan_2"])
    photoscan_filepath = example_database.get_photoscan_metadata_filepath(subject.id, session.id, "example_photoscan_2")
    assert(photoscan_filepath.name == "example_photoscan_2.json")
    assert((photoscan_filepath.parent/"example_photoscan_2.obj").exists())
    assert((photoscan_filepath.parent/"example_photoscan_texture_2.exr").exists())
