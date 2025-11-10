from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import select

from matplotlib import pyplot as plt
from openlifu.bf import apod_methods, focal_patterns, delay_methods
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup
from openlifu.sim.kwave_if import get_karray, get_medium, get_source, get_sensor, get_kgrid
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
import numpy as np
import xarray as xa
from openlifu.util.units import getunitconversion
from kwave.ksensor import kSensor
from kwave.ksource import kSource

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# set focus
simulate = True
plot = True
simulate2 = True

xInput = 0
yInput = 0
zInput = 45

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system
freq = frequency_kHz*1e3

pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(60/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=1)

here = Path(__file__).parent.resolve()
db_path = here / ".." / "OpenLIFU_Database_DCVA"
db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1_002")

simulation_options = SimulationOptions(
                        pml_auto=True,
                        pml_inside=False,
                        save_to_disk=True,
                        data_cast='single'
                    )

target = Point(position=(xInput,yInput,zInput), units="mm")

execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
spacing = 1
# spacing = 0.125
sim_setup = SimSetup(spacing=spacing, dt=2e-7, t_end=100e-6)
focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
apod_method = apod_methods.Uniform()
delay_method = delay_methods.Direct()
protocol = Protocol(
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    sim_setup=sim_setup)

pts = protocol.focal_pattern.get_targets(target)
coords = protocol.sim_setup.get_coords()
params = protocol.seg_method.ref_params(coords)
kgrid = get_kgrid(coords)

delays, apod = protocol.beamform(arr=arr, target=pts[0], params=params)

amplitude = 1
cycles = 20
t = np.arange(0, cycles / freq, kgrid.dt)
input_signal = amplitude * np.sin(2 * np.pi * freq * t)
source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)

units = [params[dim].attrs['units'] for dim in params.dims]
scl = getunitconversion(units[0], 'm')
array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]
bli_tolerance = 0.05
upsampling_rate = 5
karray = get_karray(arr,
                    translation=array_offset,
                    bli_tolerance=bli_tolerance,
                    upsampling_rate=upsampling_rate)

medium = get_medium(params)
sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
source = get_source(kgrid, karray, source_mat)

if simulate:
    output = kspaceFirstOrder3D(kgrid=kgrid,
                                    source=source,
                                    sensor=sensor,
                                    medium=medium,
                                    simulation_options=simulation_options,
                                    execution_options=execution_options)

    sz = list(params.coords.sizes.values())
    p_max = xa.DataArray(output['p_max'].reshape(sz, order='F'),
                            coords=params.coords,
                            name='p_max',
                            attrs={'units':'Pa', 'long_name':'PPP'})
    p_min = xa.DataArray(-1*output['p_min'].reshape(sz, order='F'),
                            coords=params.coords,
                            name='p_min',
                            attrs={'units':'Pa', 'long_name':'PNP'})
    Z = params['density'].data*params['sound_speed'].data
    intensity = xa.DataArray(1e-4*output['p_min'].reshape(sz, order='F')**2/(2*Z),
                            coords=params.coords,
                            name='I',
                            attrs={'units':'W/cm^2', 'long_name':'Intensity'})
    ds = xa.Dataset({'p_max':p_max, 'p_min':p_min, 'intensity':intensity})
    if plot == True:
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.colorbar()

sensor_mask_pos = np.array([el.get_position(units='m') for el in arr.elements]).T*100

if plot:
    xs, ys, zs = (sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) 
    ax.scatter(sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

x_values = np.floor(kgrid.Nx/2)
y_values = np.floor(kgrid.Ny/2)
z_values = np.floor(kgrid.Nz/2)

x_range = np.arange(-x_values/2,x_values/2+spacing,spacing)
y_range = np.arange(-y_values/2,y_values/2+spacing,spacing)
z_range = np.arange(-z_values/2,z_values/2+spacing,spacing)
ele_bin = np.zeros((kgrid.Nx,kgrid.Ny,kgrid.Nz))
ele_sensors = np.empty((128))

for i in range(128):
    ind_x = find_nearest(x_range,sensor_mask_pos[0][i])
    ind_y = find_nearest(x_range,sensor_mask_pos[1][i])
    ind_z = find_nearest(x_range,sensor_mask_pos[2][i])
    # ele_sensors[i] = np.array([ind_x,ind_y,ind_z])
    ele_bin[ind_x,ind_y,ind_z] = 1

# if plot:
#     xs, ys, zs = (ele_sensors[0],ele_sensors[1],ele_sensors[2])
#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(projection='3d')
#     ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) 
#     ax.scatter(ele_sensors[0],ele_sensors[1],ele_sensors[2])
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

# print(ele_sensors)

if simulate2:
    sensor2 = kSensor(record=['p'])
    # sensor2.mask = ele_bin
    sensor2.mask = karray.get_array_binary_mask(kgrid)
    source2 = kSource()
    source2.p0 = p_max.to_numpy()
    kgrid2 = get_kgrid(coords)
    medium2 = get_medium(params)

    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid2,
        source=source2,
        sensor=sensor2,
        medium=medium2,
        simulation_options=simulation_options,
        execution_options=execution_options
    )

    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(sensor_data['p'], aspect='auto', extent=[
            0, kgrid.Nt * kgrid.dt * 1e6,  # Time in μs
            0, sensor_data['p'].shape[0]  # Sensor number
        ])
        plt.xlabel('Time (μs)')
        plt.ylabel('Sensor Number')
        plt.title('Recorded Pressure at Boundary Sensors')
        plt.colorbar(label='Pressure (Pa)')

        plt.figure()
        plt.imshow(sensor2.mask[:,round(kgrid2.Ny/2),:])

if plot:
    plt.show()

print('num ele')
print(np.sum(ele_bin))
print(f'num grid: {np.sum(sensor2.mask)}')
# amp, phase, freq = extract_amp_phase(np.squeeze(sensor_data['p']),1/kgrid.dt,freq)
