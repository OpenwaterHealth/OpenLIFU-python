# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
from __future__ import annotations

modified_kwave_path = R'C:\Users\pjh7\git\k-wave-python'
slicer_exe = R"C:\Users\pjh7\AppData\Local\NA-MIC\Slicer 5.2.2\Slicer.exe"
import sys

sys.path.append(modified_kwave_path)
import logging

import openlifu

root = logging.getLogger()
loglevel = logging.INFO
root.setLevel(loglevel)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(loglevel)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
import numpy as np

# %%

arr = openlifu.Transducer.gen_matrix_array(nx=8, ny=8, pitch=4, kerf=.5, units="mm", impulse_response=1e5)
trans_matrix = np.array(
    [[-1,   0,  0, 0],
     [0, .05,  np.sqrt(1-.05**2), -105],
     [0, np.sqrt(1-.05**2),  -.05, 5],
     [0, 0,  0, 1]])
arr.rescale("mm")
arr.matrix = trans_matrix
pt = openlifu.Point(position=(5,-60,-8), units="mm", radius=2)

# %%
pulse = openlifu.Pulse(frequency=400e3, duration=3/400e3)
sequence = openlifu.Sequence()
focal_pattern = openlifu.focal_patterns.Wheel(center=True, spoke_radius=5, num_spokes=5)
sim_setup = openlifu.SimSetup(dt=2e-7, t_end=100e-6)
protocol = openlifu.Protocol(
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    sim_setup=sim_setup)
pts = protocol.focal_pattern.get_targets(pt)
coords = protocol.sim_setup.get_coords()
params = protocol.seg_method.ref_params(coords)
delays, apod = protocol.beamform(arr=arr, target=pts[0], params=params)


# %%
ds = openlifu.sim.run_simulation(arr=arr,
        params=params,
        delays=delays,
        apod= apod,
        freq = pulse.frequency,
        cycles = np.max([np.round(pulse.duration * pulse.frequency), 20]),
        dt=protocol.sim_setup.dt,
        t_end=protocol.sim_setup.t_end,
        amplitude = 1)

# %%
ds['p_max'].sel(lat=-5).plot.imshow()

# %%
# Export to .nii.gz
import nibabel as nb

output_filename = "foo.nii.gz"
trans_matrix = np.array(
    [[-1,   0,  0, 0],
     [0, .05,  np.sqrt(1-.05**2), -105],
     [0, np.sqrt(1-.05**2),  -.05, 5],
     [0, 0,  0, 1]])
da = ds['p_max'].interp({'lat':np.arange(-30, 30.1, 0.5),'ele':np.arange(-30, 30.1, 0.5), 'ax': np.arange(-4,70.1,0.5)})
origin_local = [float(val[0]) for dim, val in da.coords.items()]
dx = [float(val[1]-val[0]) for dim, val in da.coords.items()]
affine = np.array([-1,-1,1,1]).reshape(4,1)*np.concatenate([trans_matrix[:,:3], trans_matrix @ np.array([*origin_local, 1]).reshape([4,1])], axis=1)*np.array([*dx, 1]).reshape([1,4])
data = da.data
im = nb.Nifti1Image(data, affine)
h = im.header
h.set_xyzt_units('mm', 'sec')
im = nb.as_closest_canonical(im)
im.to_filename(output_filename)


# %%
# Load into Slicer
import slicerio.server

slicerio.server.file_load(output_filename, slicer_executable=slicer_exe)
