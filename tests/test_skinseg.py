from typing import Tuple

import numpy as np

from openlifu.seg.skinseg import (
    compute_foreground_mask,
    take_largest_connected_component,
)


def add_ball(
    volume:np.ndarray,
    center:Tuple[float,float,float],
    radius:float,
    value:float = 1.
):
    """Add a ball to a given 3D volume. Useful for creating test volumes."""
    z_size, y_size, x_size = volume.shape

    z, y, x = np.ogrid[:z_size, :y_size, :x_size]

    dist_sq = (
        (z - center[0])**2 +
        (y - center[1])**2 +
        (x - center[2])**2
    )

    volume[dist_sq <= radius**2] = value

def test_take_largest_connected_component():
    vol_array = np.zeros((20,20,20))
    add_ball(vol_array, (5,5,5), 4) # ball of radius 4 at (5,5,5)
    expected_output = np.copy(vol_array).astype(bool) # at the end we expect to only get this first ball
    add_ball(vol_array, (15,15,15), 2) # smaller ball over at (15,15,15)
    assert np.all(
        take_largest_connected_component(vol_array) == expected_output
    )

def test_compute_foreground_mask():
    vol_array = np.zeros((20,20,20))
    add_ball(vol_array, (6,6,6), 5)
    expected_output = np.copy(vol_array).astype(bool) # this first ball shall be the expected output
    add_ball(vol_array, (6,5,5), 2, value=0.) # erase a hole inside this first ball
    add_ball(vol_array, (15,15,15), 3) # make a smaller disconnected ball elsewhere
    assert np.all(
        compute_foreground_mask(vol_array) == expected_output
    )
