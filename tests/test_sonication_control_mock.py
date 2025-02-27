from datetime import datetime

import numpy as np
import pytest
import xarray as xa

from openlifu.bf import Pulse, Sequence
from openlifu.geo import Point
from openlifu.io.LIFUInterface import STATUS_READY, LIFUInterface
from openlifu.plan.solution import Solution


@pytest.fixture()
def example_solution() -> Solution:
    rng = np.random.default_rng(147)
    return Solution(
        id="sol_001",
        name="Test Solution",
        protocol_id="prot_123",
        transducer_id="trans_456",
        date_created=datetime(2024, 1, 1, 12, 0),
        description="This is a test solution for a unit test.",
        delays=np.array([[0.0, 1.0, 2.0, 3.0]]),
        apodizations=np.array([[0.5, 0.75, 1.0, 0.85]]),
        pulse=Pulse(frequency=42),
        sequence=Sequence(pulse_count=27),
        foci=[Point(id="test_focus_point")],
        target=Point(id="test_target_point"),
        simulation_result=xa.Dataset(
            {
                'p_min': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'p_max': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'ita': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "W/cm^2"}
                )
            },
            coords={
                'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
                'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'focal_point_index': [0]
            }
        ),
    )

def test_lifuinterface_mock(example_solution:Solution):
    """Test that LIFUInterface can be used in mock mode (i.e. test_mode=True)"""
    lifu_interface = LIFUInterface(test_mode=True)
    lifu_interface.set_solution(example_solution)
    lifu_interface.start_sonication()
    status = lifu_interface.get_status()
    assert status == STATUS_READY
    lifu_interface.stop_sonication()
