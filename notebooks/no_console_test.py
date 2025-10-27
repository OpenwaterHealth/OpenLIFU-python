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
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan import Protocol
from openlifu.sim import SimSetup

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

log_interval = 1  # seconds; you can adjust this variable as needed

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
delay_method = delay_methods.TimeReversal()
# apod_method = apod_methods.MaxAngle()
# apod_method = apod_methods.PiecewiseLinear()
sim_setup = SimSetup(x_extent=(-55,55), y_extent=(-30,30), z_extent=(-4,150))
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
voltage = solution.voltage
peak_to_peak_voltage = solution.voltage * 2 # Peak to peak voltage for the pulse

logger.info(f'Apodizations: {solution.apodizations}')
logger.info(f'Delays: {solution.delays}')

profile_index = 1
profile_increment = True
trigger_mode = "continuous"

duty_cycle = int((duration_msec/interval_msec) * 100)
if duty_cycle > 50:
    logger.warning("❗❗ Duty cycle is above 50% ❗❗")

logger.info(f"User parameters set: \n\
    Module Invert: {arr.module_invert}\n\
    Frequency: {frequency_kHz}kHz\n\
    Voltage Per Rail: {solution.voltage}V\n\
    Voltage Peak to Peak: {peak_to_peak_voltage}V\n\
    Duration: {duration_msec}ms\n\
    Interval: {interval_msec}ms\n\
    Duty Cycle: {duty_cycle}%\n\
    Use External Power Supply: {use_external_power_supply}\n\
    Initial Temp Safety Shutoff: Increase to {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s of starting.\n\
    General Temp Safety Shutoff: Increase of {rapid_temp_increase_per_second_shutoff_C}°C within {log_interval}s at any point.\n")


p_map = sim_res['p_max']
if scaled_analysis is not None:
    scaled_analysis.to_table().set_index('Param')[['Value', 'Units', 'Status']]

plt.imshow(p_map[:,30,:])
plt.colorbar()
plt.show()

