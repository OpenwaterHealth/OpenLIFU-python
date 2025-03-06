from __future__ import annotations

import numpy as np

from openlifu.geo import (
    Point,
    cartesian_to_spherical,
    cartesian_to_spherical_vectorized,
    spherical_coordinate_basis,
    spherical_to_cartesian,
    spherical_to_cartesian_vectorized,
)


def test_point_from_dict():
    point = Point.from_dict({'position' : [10,20,30],})
    assert (point.position == np.array([10,20,30], dtype=float)).all()

def test_spherical_coordinate_range():
    """Verify that spherical coordinate output is in the prescribed value ranges"""
    rng = np.random.default_rng(848)
    # try all 8 octants of 3D space
    for sign_x in [-1,1]:
        for sign_y in [-1,1]:
            for sign_z in [-1,1]:
                cartesian_coords = np.array([sign_x, sign_y, sign_z]) * rng.random(size=3)
                r, th, ph = cartesian_to_spherical(*cartesian_coords)
                assert r>=0
                assert 0 <= th <= np.pi
                assert -np.pi <= ph <= np.pi

def test_spherical_coordinate_conversion_inverse():
    """Verify that the spherical coordinate conversion forward and backward functions are inverses of one another"""
    rng = np.random.default_rng(241)
    # try all 8 octants of 3D space
    for sign_x in [-1,1]:
        for sign_y in [-1,1]:
            for sign_z in [-1,1]:
                cartesian_coords = np.array([sign_x, sign_y, sign_z]) * rng.random(size=3)
                np.testing.assert_almost_equal(
                    spherical_to_cartesian(*cartesian_to_spherical(*cartesian_coords)),
                    cartesian_coords
                )
                np.testing.assert_almost_equal(
                    cartesian_to_spherical(*spherical_to_cartesian(*cartesian_to_spherical(*cartesian_coords))),
                    cartesian_to_spherical(*cartesian_coords)
                )

def test_cartesian_to_spherical_vectorized():
    rng = np.random.default_rng(35932)
    points_cartesian = rng.normal(size=(10,3), scale=2) # make 10 random cartesian points
    points_spherical = cartesian_to_spherical_vectorized(points_cartesian)
    # Check individual points against the non-vectorized conversion function:
    for point_cartesian, point_spherical in zip(points_cartesian, points_spherical):
        assert np.allclose(
            point_spherical, # result of vectorized converter
            np.array(cartesian_to_spherical(*point_cartesian)), # non-vectorized converter
        )

def test_spherical_to_cartesian_vectorized():
    rng = np.random.default_rng(85932)

    # make 10 random points in spherical coordinates
    num_pts = 10
    points_spherical = np.zeros(shape=(num_pts,3))
    points_spherical[...,0] = rng.random(num_pts)*5 # random r coordinates
    points_spherical[...,1] = rng.random(num_pts)*np.pi # random theta coordinates
    points_spherical[...,2] = rng.random(num_pts)*2*np.pi-np.pi # random phi coordinates

    points_cartesian = spherical_to_cartesian_vectorized(points_spherical)
    # Check individual points against the non-vectorized conversion function:
    for point_cartesian, point_spherical in zip(points_cartesian, points_spherical):
        assert np.allclose(
            point_cartesian, # result of vectorized converter
            np.array(spherical_to_cartesian(*point_spherical)), # non-vectorized converter
        )

def test_spherical_coordinate_basis():
    rng = np.random.default_rng(35235)
    th = rng.random()*np.pi
    phi = rng.random()*2*np.pi-np.pi
    r  = rng.random()*10
    basis = spherical_coordinate_basis(th,phi)
    assert np.allclose(basis @ basis.T, np.eye(3)) # verify it is an orthonormal basis
    r_hat, theta_hat, phi_hat = basis
    point = np.array(spherical_to_cartesian(r, th, phi))
    assert np.allclose(np.diff(r_hat / point), 0) # verify that r_hat is a scalar multiple of the cartesian coords
    assert cartesian_to_spherical_vectorized(point + 0.01*phi_hat)[2] > phi # verify phi_hat points along increasing phi
    assert cartesian_to_spherical_vectorized(point + 0.01*theta_hat)[1] > th # verify theta_hat points along increasing theta
