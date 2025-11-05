import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from openlifu.bf import apod_methods, focal_patterns, delay_methods
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan import Protocol
from openlifu.sim import SimSetup


width = 56
height = 56
depth = 56

x = np.arange(width)
y = np.arange(height)
z = np.arange(depth)

X, Y, Z = np.meshgrid(x, y, z)
p = np.random.rand(width,height,depth)
noise = np.random.normal(p,0.2*p)
wavelength_x = 100000 
wavelength_y = 15000 
amplitude = 1

sine_image = amplitude * np.sin(2 * np.pi * (X / wavelength_x + Y / wavelength_y)) + p

sine_image = np.abs(sine_image)
max_val = sine_image.max()
scale_factor = 1500/max_val
sine_image = sine_image*scale_factor
sine_image = ndimage.median_filter(sine_image,3)

plt.figure()
plt.imshow(sine_image[:,:,10])
plt.colorbar()
plt.show()

# medium parameters
c_min               = 1500     # sound speed [m/s]
c_max               = 3100     # max. speed of sound in skull (F. A. Duck, 2013.) [m/s]
rho_min             = 1000     # density [kg/m^3]
rho_max             = 1900     # max. skull density [kg/m3]
alpha_power         = 1.43     # Robertson et al., PMB 2017 usually between 1 and 3? from Treeby paper
alpha_coeff_water   = 0        # [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017), see also 0.05 Fomenko et al., 2020?
alpha_coeff_min     = 4     
alpha_coeff_max     = 8.7      # [dB/(MHz cm)] Fry 1978 at 0.5MHz: 1 Np/cm (8.7 dB/cm) for both diploe and outer tables

hu_min 	= 300
hu_max 	= 2000	

if max_val < hu_max:
    hu_max = max_val

sine_image[sine_image<hu_min] = 0
sine_image[sine_image>hu_max] = hu_max
padx = 20
tmp_model = np.zeros(np.size(sine_image))

# set focus
xInput = 20
yInput = 0
zInput = 80

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system

use_external_power_supply = False # Select whether to use console or power supply

console_shutoff_temp_C = 70.0 # Console shutoff temperature in Celsius
tx_shutoff_temp_C = 70.0 # TX device shutoff temperature in Celsius
ambient_shutoff_temp_C = 70.0 # Ambient shutoff temperature in Celsius

#TODO: script_timeout_minutes = 30 # Prevent unintentionally leaving unit on for too long
#TODO: log_temp_to_csv_file = True # Log readings to only terminal or both terminal and CSV file

# Fail-safe parameters if the temperature jumps too fast
rapid_temp_shutoff_C = 40 # Cutoff temperature in Celsius if it jumps too fast
rapid_temp_shutoff_seconds = 5 # Time in seconds to reach rapid temperature shutoff
rapid_temp_increase_per_second_shutoff_C = 3 # Rapid temperature climbing shutoff in Celsius


here = Path(__file__).parent.resolve()
db_path = here / ".." / "OpenLIFU_Database_DCVA"
db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1_002")

arr.sort_by_pin()

target = Point(position=(xInput,yInput,zInput), units="mm")

pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(60/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=1)

focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
apod_method = apod_methods.Uniform()
delay_method = delay_methods.Direct()

midpoint = np.round([xInput/2,yInput/2,zInput/2])

x_extent=(-55,55)
y_extent=(-30,30)
z_extent=(-4,150)

padx = np.round((np.abs(x_extent[0]) + np.abs(x_extent[1]) - width)/2)
pady = np.round((np.abs(y_extent[0]) + np.abs(y_extent[1]) - height)/2)
padz = np.round((np.abs(z_extent[0]) + np.abs(z_extent[1]) - depth)/2)

tmp = np.zeros(np.size(sine_image,0)+padx,np.size(sine_image,1)+pady,np.size(sine_image,2)+padz)
print(np.size(tmp))
sim_setup = SimSetup(x_extent, y_extent, z_extent)

protocol = Protocol(
    id='test_protocol',
    name='Test Protocol',
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    apod_method=apod_method,
    sim_setup=sim_setup)

solution, sim_res, scaled_analysis = protocol.calc_solution(
    target=target,
    transducer=arr,
    simulate=True,
    scale=True,
    use_gpu=True)

p_map = sim_res['p_max']
plt.figure()
plt.imshow(p_map[:,30,:])
plt.colorbar()
plt.show()


