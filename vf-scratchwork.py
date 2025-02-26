# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
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
# VF Algorithm Inputs

target_RAS = (1.268, 52.346, 9.786)
pitch_range = (-10,40) # in deg
pitch_step = 3 # in deg
yaw_range = (-5,25) # in deg
yaw_step = 2 # in deg
radius_in_mm = 50
steering_limits = [ # Should really be a List[TargetConstraints] which initializes to empty
    TargetConstraints(dim="lat", units="mm", min=-100, max=100),
    TargetConstraints(dim="ele", units="mm", min=-100, max=100),
    TargetConstraints(dim="ax",  units="mm", min=0, max=300),
]
blocked_elems_threshold = 0.1
volume_array = vol_array
volume_affine_RAS = vol_affine # We assume this maps into RAS space! That's how nifti works.
# scene_matrix # I don't see where this is even used.
# scene_origin # I don't see where this is used either
transducer = Transducer.from_file("db_dvc/transducers/curved2_100roc_10gap/curved2_100roc_10gap.json")

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

# The goal here is to get the "transducer pose" which means a 4x4 (or 3x4) matrix that would work as a "transducer transform", i.e.
# maps the transducer into LPS volume space
# TODO: FIX the below nonsense -- I need to read more carefully what they are doing.
# To do this, we first build a fitting grid of points in the $\partial\theta,\partial\phi$ (coordinate basis) plane at the point.
# (Wait, why. Why not just a theta and phi grid
# Then we pass those points through the
