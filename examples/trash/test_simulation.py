from __future__ import annotations

import os
from pathlib import Path

if os.name == 'nt':
    pass
else:
    pass

import openlifu
from openlifu.bf import apod_methods, focal_patterns
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup

xInput = 0
yInput = 0
zInput = 30

frequency_kHz = 150 # Frequency in kHz
duration_msec = 100 # Pulse Duration in milliseconds
interval_msec = 200 # Pulse Repetition Interval in milliseconds
num_modules = 1 # Number of modules in the system

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
db_path = Path(openlifu.__file__).parent.parent.parent / "db_dvc"
db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1")

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
sim_setup = SimSetup(x_extent=(-55,55), y_extent=(-30,30), z_extent=(-4,70))
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
    voltage=65,
    simulate=True,
    scale=False)

print(solution.analyze().to_table())
