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
def create_standoff_transform(z_offset:float, dzdy:float) -> np.ndarray:
    """Create a standoff transform based on a z_offset and a dzdy value.
    
    A "standoff transform" applies a displacement in transducer space that moves a transducer to where it would
    be situated with the standoff in place. The idea is that if you start with a transform that places a transducer
    directly against skin, then pre-composing that transform by a "standoff transform" serves to nudge the transducer
    such that there is space for the standoff to be between it and the skin.

    This function assumes that the standoff is laterally symmetric, has some thickness, and can raise the bottom of
    the transducer a bit more than the top. The `z_offset` is the thickness in the middle of the standoff,
    while the `dzdy` is the elevational slope.

    Args:
        z_offset: Thickness in the middle of the standoff
        dzdy: Slope of the standoff, as axial displacement per unit elevational displacement. A positive number
            here means that the bottom of the transducer is raised a little bit more than the top.

    Returns a 4x4 matrix representing a rigid transform in whatever units z_offset was provided in.
    """
    angle = np.arctan(dzdy)
    return np.array([
        [1,0,0,0],
        [0,np.cos(angle),-np.sin(angle),0],
        [0,np.sin(angle),np.cos(angle),-z_offset],
        [0,0,0,1],
    ], dtype=float)

def test_create_standoff_transform():
    z_offset = 3.2
    dzdy = 0.15
    t = create_standoff_transform(z_offset, dzdy)
    assert np.allclose(t[:3,:3] @ t[:3,:3].T, np.eye(3)) # it's an orthonormal transform
    assert np.allclose(np.linalg.det(t[:3,:3]), 1.0) # orientation preserving
    assert np.allclose(t @ np.array([0,0,0,1]), np.array([0,0,-z_offset,1.])) # translates the origin correctly
    new_x_axis = (t @ np.array([1,0,0,1]) - t @ np.array([0,0,0,1]))[:3]
    new_y_axis = (t @ np.array([0,1,0,1]) - t @ np.array([0,0,0,1]))[:3]
    assert np.allclose(new_x_axis, np.array([1.,0,0]))
    assert new_y_axis[2] > 0 # the y axis was rotated upward, so that the top of the transducer gets closer to the skin

test_create_standoff_transform()

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
volume_array = vol_array # TODO: My feeling is that the units of the volume array (or rather of the RAS space that it is mapped to via volume_affine_RAS) need to be provided since everything ends up in those units.
volume_affine_RAS = vol_affine # We assume this maps into RAS space! That's how nifti works. # TODO: are we implicitly assuming both transducer and volume are in same units? are we assuming it's mm anywhere?
# scene_matrix # I don't see where this is even used.
# scene_origin # I don't see where this is used either
transducer = Transducer.from_file("db_dvc/transducers/curved2_100roc_10gap/curved2_100roc_10gap.json")

# Note that these are in the units of the volume space! I think.
planefit_dyaw_extent = 20 # we go out as far as +/- extent, so the search space is twice this size
planefit_dyaw_step = 5
planefit_dpitch_extent = 15
planefit_dpitch_step = 1

standoff_transform = create_standoff_transform(
    z_offset = 13.55, # units of the *transducer*!
    dzdy = 0.15
) # therefore this transform is units of the *transducer*

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
# - Many operations are now vectorized rather than relying on a python loop. There is still a python loop over the virtual fit search grid points, but there is no inner python loop over the plane-fitting points.

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
# Visualize entire virtual fitting search grid
pts = []
for i in range(num_thetas):
    for j in range(num_phis):
        theta_rad, phi_rad = theta_grid[i,j]*np.pi/180, phi_grid[i,j]*np.pi/180
        pts.append(spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad))
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=pts, camera_focus=(0,0,0))

# %%
# We will iterate over all i and j as in the "visualize search grid" cell above, but in this cell we just demo on one grid point i=0,j=0
theta_rad, phi_rad = theta_grid[0,0]*np.pi/180, phi_grid[0,0]*np.pi/180

# Cartesian coordinate location of the point at which we are fitting a plane
point = np.array(spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad))

# The point at which we are now fitting a plane
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad), camera_focus=(0,0,0))

# Build plane fitting grid in the spherical coordinate basis theta-phi plane, which we will later project back onto the skin surface
dtheta_sequence = np.arange(-planefit_dyaw_extent, planefit_dyaw_extent + planefit_dyaw_step, planefit_dyaw_step)
dphi_sequence = np.arange(-planefit_dpitch_extent, planefit_dpitch_extent + planefit_dpitch_step, planefit_dpitch_step)
dtheta_grid, dphi_grid = np.meshgrid(dtheta_sequence, dphi_sequence, indexing='ij')

r_hat, theta_hat, phi_hat = spherical_coordinate_basis(theta_rad,phi_rad)
planefit_points_unprojected_cartesian = (
    point.reshape(1,1,3)
    + dtheta_grid[...,np.newaxis] * theta_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
    + dphi_grid[...,np.newaxis] * phi_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
) # shape (num dthetas, num dphis, 3)

# visualize unprojected plane fitting points alongside sphere of radius `point`'s r-coordinate
# visualize_polydata(sphere_from_interpolator(lambda x,y : np.sqrt(np.sum(point**2))), highlight_points=list(planefit_points_unprojected_cartesian.reshape(-1,3)), camera_focus=(0,0,0))

# visualize unprojected plane fitting points alongside the actual skin surface
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=list(planefit_points_unprojected_cartesian.reshape(-1,3)), camera_focus=(0,0,0))

planefit_points_unprojected_spherical = cartesian_to_spherical_vectorized(
    planefit_points_unprojected_cartesian
) # shape (num dthetas, num dphis, 3)
skin_projected_r_values = skin_interpolator(planefit_points_unprojected_spherical[...,1:]) # shape (num dthetas, num dphis) # TODO adjust docstrings to demand a *vectorizable* spherical interpolator
planefit_points_cartesian = spherical_to_cartesian_vectorized( # Could instead renormalize planefit_points_unprojected_cartesian, not sure if it would give a speedup versus this
    np.stack([
        skin_projected_r_values, # New r values after projection to skin
        planefit_points_unprojected_spherical[...,1], # Same old theta values
        planefit_points_unprojected_spherical[...,2], # Same old phi values
    ], axis=-1)
)

# visualize the plane fitting points now projected onto the skin surface
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=list(planefit_points_cartesian.reshape(-1,3)), camera_focus=(0,0,0))

# Fit the best plane to these points among the planes that pass through `point`. Here we find the normal vector to the plane.
plane_normal = np.linalg.svd(
    planefit_points_cartesian.reshape(-1,3)-point.reshape(1,3),
    full_matrices=False, # we don't need the left-singular vectors anyway, so this speeds things up
).Vh[-1] # The right-singular vector corresponding to the smallest singular value

# visualize the fitted plane
# visualize_polydata(sphere_from_interpolator(skin_interpolator), highlight_points=list(planefit_points_cartesian.reshape(-1,3)), camera_focus=point, additional_actors=[create_plane_actor(point, plane_normal, plane_size=30)])

# NEXT: Define standoff_transform based on z_offset and dzdy way up in the inputs, and make the standoff_transform be the actual input.
# An add an issue, or add to the existing issue on steering limits, that standoff_transform should be a transducer attribute.

# Now that we have fit a plane for the transducer to go onto, we place the transducer so that
# (1) the transducer axial (z) axis is parallel to plane_normal and points towards the origin rather than away from it (dot product with `point` is negative)
# (2) the transducer elevational (y) axis is pretty much along phi-hat but must be orthogonal to plane_normal. we can start with phi-hat and subtract its component along plane_normal and normalize.
# (3) the transducer lateral (x) coordinate is simply y cross z.
# This all can be put together to give a 4x4 transducer transform matrix that places the transducer into this ASL space in the units of the volume and with the target as the origin.
# We can multiply it by a fixed matrix that we have that makes it into an LPS space transform with the original origin of the volume array, the units still being the units of the volume array, then we can return it.
# And finally we can precompose the matrix with the fixed standoff_transform
# Yeah, so this will be a function here, that goes from theta_rad, phi_rad and w/e other parameters to give you the transducer transform matrix in the volume's units.
# Then we build out the loop to build up a collection of transducer poses using this function, and we proceed to port over what is in the virtual fit algorithm by loic and tong.


# %%
