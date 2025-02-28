from __future__ import annotations

from pathlib import Path

import pytest
from helpers import dataclasses_are_equal

from openlifu.plan import Run


@pytest.fixture()
def example_run() -> Run:
    return Run(
        id="example_run",
        name="example_run_name",
        success_flag = True,
        note="Example note",
        session_id="example_session",
        solution_id="example_solution",
    )

def test_default_run():
    """Ensure it is possible to construct a default Run"""
    Run()

def test_save_load_run_from_file(example_run:Run, tmp_path:Path):
    """Test that a run can be saved to and loaded from disk faithfully."""
    json_filepath = tmp_path/"some_directory"/"example_run.json"
    example_run.to_file(json_filepath)
    assert dataclasses_are_equal(example_run.from_file(json_filepath), example_run)

@pytest.mark.parametrize("compact_representation", [True, False])
def test_save_load_run_from_json(example_run:Run, compact_representation: bool):
    """Test that a run can be saved to and loaded from json faithfully."""
    run_json = example_run.to_json(compact = compact_representation)
    assert dataclasses_are_equal(example_run.from_json(run_json), example_run)
