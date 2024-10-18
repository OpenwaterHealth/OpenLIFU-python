from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xa
from helpers import dataclasses_are_equal

from openlifu import Point, Pulse, Sequence, Solution, Transducer
from openlifu.xdc.element import Element


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer(
        id="trans_456",
        name="Test Transducer",
        elements=[
            Element(index=1, x=-14, y=-14, units="m"),
            Element(index=2, x=-2, y=-2, units="m"),
            Element(index=3, x=2, y=2, units="m"),
            Element(index=4, x=14, y=14, units="m")
        ],
        frequency=1e6,
        units="m"
    )


@pytest.fixture()
def example_solution() -> Solution:
    rng = np.random.default_rng(147)
    return Solution(
        id="sol_001",
        name="Test Solution",
        protocol_id="prot_123",
        transducer_id="trans_456",
        created_on=datetime(2024, 1, 1, 12, 0),
        description="This is a test solution for a unit test.",
        delays=np.array([0.0, 1.0, 2.0, 3.0]),
        apodizations=np.array([0.5, 0.75, 1.0, 0.85]),
        pulse=Pulse(frequency=42),
        sequence=Sequence(pulse_count=27),
        focus=Point(id="test_focus_point"),
        target=Point(id="test_target_point"),
        simulation_result=xa.Dataset(
            {
                'p_min': xa.DataArray(
                    data=rng.random((3, 2, 3)),
                    dims=["x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'p_max': xa.DataArray(
                    data=rng.random((3, 2, 3)),
                    dims=["x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'ita': xa.DataArray(
                    data=rng.random((3, 2, 3)),
                    dims=["x", "y", "z"],
                    attrs={'units': "W/cm^2"}
                )
            },
            coords={
                'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
                'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"})
            }
        ),
    )


def test_default_solution():
    """Ensure it is possible to construct a default Solution"""
    Solution()


@pytest.mark.parametrize("compact_representation", [True, False])
@pytest.mark.parametrize("include_simulation_data", [True, False])
@pytest.mark.parametrize("default_solution", [True, False])
def test_json_serialize_deserialize_solution(
    example_solution: Solution,
    compact_representation: bool,
    include_simulation_data: bool,
    default_solution: bool
):
    """Verify that turning a Solution into json and then re-constructing it gets back to the original solution"""

    # Default solution serializes a bit differently because it's full of None values, so we test both cases
    solution = Solution() if default_solution else example_solution
    solution_json = solution.to_json(include_simulation_data=include_simulation_data, compact=compact_representation)
    if include_simulation_data:
        solution_reconstructed = Solution.from_json(solution_json)
    else:
        solution_reconstructed = Solution.from_json(solution_json, simulation_result=solution.simulation_result)
    assert dataclasses_are_equal(solution_reconstructed, solution)


def test_save_load_solution(example_solution: Solution, tmp_path: Path):
    """Test that a solution can be saved to and loaded from disk faithfully."""
    json_filepath = tmp_path/"some_directory"/"example_soln.json"
    example_solution.to_files(json_filepath)
    assert dataclasses_are_equal(Solution.from_files(json_filepath), example_solution)


def test_save_load_solution_custom_dataset_filepath(example_solution: Solution, tmp_path: Path):
    """Test that a solution can be saved to and loaded from disk faithfully, this time with a custom path for simulation data."""
    json_filepath = tmp_path/"some_directory"/"example_soln.json"
    nc_filepath = tmp_path/"some_other_directory"/"sim_output.nc"
    example_solution.to_files(json_filepath, nc_filepath)
    assert dataclasses_are_equal(Solution.from_files(json_filepath, nc_filepath), example_solution)


def test_solution_analysis(example_solution: Solution, example_transducer: Transducer):
    """Test that a solution output can be analyzed."""
    example_solution.analyze(example_transducer)
