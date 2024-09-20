from datetime import datetime

import numpy as np
import pytest
import xarray
from helpers import dataclasses_are_equal

from openlifu import Point, Pulse, Sequence, Solution


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
        focus=Point(id = "test_focus_point"),
        target=Point(id = "test_target_point"),
        simulation_result=xarray.Dataset(
            {'pnp': (['x', 'y', 'z'], rng.random((3, 2, 3)))},
            coords={'x': np.linspace(0, 1, 3), 'y': np.linspace(0, 1, 2), 'z': np.linspace(0, 1, 3)},
        ),
    )

def test_default_solution():
    """Ensure it is possible to construct a default Solution"""
    Solution()

@pytest.mark.parametrize("compact_representation", [True, False])
@pytest.mark.parametrize("include_simulation_data", [True, False])
@pytest.mark.parametrize("default_solution", [True, False])
def test_json_serialize_deserialize_solution(
    example_solution:Solution,
    compact_representation:bool,
    include_simulation_data:bool,
    default_solution:bool
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
