# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
from viz import *

from openlifu.plan import TargetConstraints
from openlifu.seg.skinseg import (
    cartesian_to_spherical,
    compute_foreground_mask,
    create_closed_surface_from_labelmap,
    spherical_interpolator_from_mesh,
    spherical_to_cartesian,
    vtk_img_from_array_and_affine,
)
from openlifu.xdc import Transducer

# %%
vol_path = Path("./MRHead.nii.gz")
nifti_img = nib.load(vol_path)
vol_array = nifti_img.get_fdata()
vol_affine = nifti_img.affine


# %%
# Utilities needed

def cartesian_to_spherical_vectorized(p:np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates to spherical coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point cartesian coordinates x,y,z.
    Returns: An array of shape (...,3), where te last axis describes point spherical coordinates r, theta, phi, where
        r is the radial spherical coordinate, a nonnegative float.
        theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle.
            theta is in the range [0,pi].
        phi is the azimuthal spherical coordinate, in the range [-pi,pi]

    Angles are in radians.
    """
    return np.stack([
        np.sqrt((p**2).sum(axis=-1)),
        np.arctan2(np.sqrt((p[...,0:2]**2).sum(axis=-1)),p[...,2]),
        np.arctan2(p[...,1],p[...,0]),
    ], axis=-1)

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

test_cartesian_to_spherical_vectorized()


# %%
def spherical_to_cartesian_vectorized(p:np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point spherical coordinates r, theta, phi, where:
            r is the radial spherical coordinate
            theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle
            phi is the azimuthal spherical coordinate
    Returns the cartesian coordinates x,y,z

    Angles are in radians.
    """
    return np.stack([
        p[...,0]*np.sin(p[...,1])*np.cos(p[...,2]),
        p[...,0]*np.sin(p[...,1])*np.sin(p[...,2]),
        p[...,0]*np.cos(p[...,1]),
    ], axis=-1)

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

test_spherical_to_cartesian_vectorized()


# %%
def spherical_coordinate_basis(th:float, phi:float) -> np.ndarray:
    """Return normalized spherical coordinate basis at a location with spherical polar and azimuthal coordinates (th, phi).
    The coordinate basis is returned as an array `basis` of shape (3,3), where the rows are the basis vectors,
    in the order r, theta, phi. So `basis[0], basis[1], basis[2]` are the vectors $\hat{r}$, $\hat{\theta}$, $\hat{\phi}$.
    Angles are assumed to be provided in radians."""
    return np.array([
        [np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)],
        [np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0],
    ])

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

test_spherical_coordinate_basis()

# %%
# VF Algorithm Inputs

target_RAS = (1.268, 52.346, 9.786)
pitch_range = (-10,40) # in deg
pitch_step = 3 # in deg
yaw_range = (-5,25) # in deg
yaw_step = 2 # in deg
radius_in_mm = 50
steering_limits = [ # Should really be a List[TargetConstraints] which initializes to empty
    TargetConstraints(dim="lat", units="NONE", min=-100, max=100), # TODO: remove units from TargetConstraints since it doesn't apply them
    TargetConstraints(dim="ele", units="NONE", min=-100, max=100), # (so the units would be the units of the transducer? I guess?)
    TargetConstraints(dim="ax",  units="NONE", min=0, max=300),
]
blocked_elems_threshold = 0.1
volume_array = vol_array
volume_affine_RAS = vol_affine # We assume this maps into RAS space! That's how nifti works. # TODO: are we implicitly assuming both transducer and volume are in same units? are we assuming it's mm anywhere?
# scene_matrix # I don't see where this is even used.
# scene_origin # I don't see where this is used either
transducer = Transducer.from_file("db_dvc/transducers/curved2_100roc_10gap/curved2_100roc_10gap.json")

# Note that these are in the units of the volume space! I think.
planefit_dyaw_extent = 20
planefit_dyaw_step = 1
planefit_dpitch_extent = 15
planefit_dpitch_step = 1

# Desired output from this will be:
# A transducer transform (an ArrayTransform) that places the transducer into the MRI space in LPS coordinates
# (units probably mm, but they are specified in the ArrayTransform so w/e)

# %%
foreground_mask_array = compute_foreground_mask(volume_array)
foreground_mask_vtk_image = vtk_img_from_array_and_affine(foreground_mask_array, volume_affine_RAS)
skin_mesh = create_closed_surface_from_labelmap(foreground_mask_vtk_image)

# %%
skin_interpolator = spherical_interpolator_from_mesh(
    surface_mesh = skin_mesh,
    origin = target_RAS,
    xyz_direction_columns = np.array([[0,1,0],[0,0,1],[-1,0,0]], dtype=float).T, # ASL, Anterior-Superior-Left, coordinates. From RAS.
)

# %%
# visualize_polydata(skin_mesh, highlight_points=target_RAS, camera_focus=target_RAS)

# %%
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=(0,0,0), camera_focus=(0,0,0))

# %% [markdown]
# Running list of major changes
#
# - The input volume is now just given as a numpy array with an affine, not an xarray thing. This is more like how a user would actually have a volume after reading it from disk.
# - (internal change) yaw and pitch are internally converted to spherical coordinates $\theta$ and $\phi$ to make it easier to work in that coordinate system. Also $\theta$ and $\phi$ were swapped from the minor previously existing usage, to match the physics convention. The relation is that pitch is $\phi$ and yaw is $90-\theta$. The $90-\theta$ is there to allow us to use a more standard version of spherical coordinates (where previously we had a confusing swap of sines with cosines from the standard approach).

# %%
# Construct search grid

theta_sequence = np.arange(90 - yaw_range[-1], 90 - yaw_range[0], yaw_step)
phi_sequence = np.arange(pitch_range[0], pitch_range[-1], pitch_step)
theta_grid, phi_grid = np.meshgrid(theta_sequence, phi_sequence, indexing="ij") # each has shape (number of thetas, number of phis)
search_grid_shape = theta_grid.shape
num_thetas, num_phis = search_grid_shape

# %%
transducer_poses = np.empty(search_grid_shape+(4,4), dtype=float)
in_bounds = np.zeros(shape=search_grid_shape, dtype=bool)
steering_dists = np.zeros(shape=search_grid_shape, dtype=float)

# %%
# Visualize search grid
pts = []
for i in range(num_thetas):
    for j in range(num_phis):
        theta_rad, phi_rad = theta_grid[i,j]*np.pi/180, phi_grid[i,j]*np.pi/180
        pts.append(spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad))
visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=pts, camera_focus=(0,0,0))

# %%
# We will iterate over all i and j as in the "visualize search grid" cell above, but in this cell we just demo on one grid point i=0,j=0
theta_rad, phi_rad = theta_grid[0,0]*np.pi/180, phi_grid[0,0]*np.pi/180

# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad), camera_focus=(0,0,0))

# To radians
dtheta_extent = planefit_dyaw_extent*np.pi/180
dtheta_step = planefit_dyaw_step*np.pi/180
dphi_extent = planefit_dpitch_extent*np.pi/180
dphi_step = planefit_dpitch_step*np.pi/180

# Build plane fitting grid in the spherical coordinate basis theta-phi plane
dtheta_sequence = np.arange(-dtheta_extent, dtheta_extent + dtheta_step, dtheta_step)
dphi_sequence = np.arange(-dphi_extent, dphi_extent + dphi_step, dphi_step)
dtheta_grid, dphi_grid = np.meshgrid(dtheta_sequence, dphi_sequence, indexing='ij')

spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad)

# NEXT: vectorize cartesian/spherical converter functions, then create a coordinate basis function (vectorized or not) and unit test it

# The goal here is to get the "transducer pose" which means a 4x4 (or 3x4) matrix that would work as a "transducer transform", i.e.
# maps the transducer into LPS volume space
# To do this, we first build a fitting grid of points on the skin surface as follows:
# - Work in the spherical coordinate basis at the location on the skin surface that this theta_rad, phi_rad correspond to
# - Make an evenly spaced fitting grid in that coordinate system inside the theta and phi coordinate basis vectors
# - Map those points in the tangent space into R^3 and apply the skin interpolator to project them down onto the skin surface
# - (I think a better approach to getting those points is to apply the riemannian exponential map to a polar grid of vectors in the tangenet space
# - Then we convert these points to the coordinate basis
# Next, we fit a plane to the points
# Then we put the transducer in that plane, while also taking into account z_offset and dzdy

# %%
