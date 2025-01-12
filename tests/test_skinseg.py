from typing import Tuple

import numpy as np
import pytest
from scipy.linalg import expm

from openlifu.seg.skinseg import (
    compute_foreground_mask,
    create_closed_surface_from_labelmap,
    take_largest_connected_component,
    vtk_img_from_array_and_affine,
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

def test_vtk_img_from_array_and_affine():
    rng = np.random.default_rng(241)
    vol_array = rng.random((5,4,3))
    affine = np.eye(4)
    affine[:3,:3] = expm((lambda A: (A - A.T)/2)(rng.normal(size=(3,3)))) # generate a random orthogonal matrix
    affine[:3,3] = rng.random(3) # generate a random origin
    vtk_img = vtk_img_from_array_and_affine(vol_array, affine)

    i,j,k = 2,3,1 # We will test at the point with these indices i,j,k. The image value there should be vol_array[i,j,k].
    x,y,z,_ = affine @ np.array([i,j,k,1]) # the physical coordinates of the test point
    point_id = vtk_img.FindPoint([x,y,z])

    assert vtk_img.GetPointData().GetScalars().GetTuple1(point_id) == pytest.approx(vol_array[i,j,k])

def test_create_closed_surface_from_labelmap():
    # create a ball of radius 7 for a labelmap
    labelmap = np.zeros((20,20,20))
    sphere_radius = 7
    sphere_center = np.array([10,10,10])
    add_ball(labelmap, tuple(sphere_center), sphere_radius)
    labelmap_vtk = vtk_img_from_array_and_affine(labelmap, affine = np.eye(4))

    # run the algorithm to be tested
    surface = create_closed_surface_from_labelmap(labelmap_vtk, decimation_factor=0.1)

    # verify that the points on the generated mesh are not too far off being at distance 7 from the ball center
    points = surface.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        point_position = np.array(points.GetPoint(i))
        point_distance_from_sphere_center = np.linalg.norm(point_position - sphere_center, ord=2)
        assert np.abs(point_distance_from_sphere_center - sphere_radius) < 1.
