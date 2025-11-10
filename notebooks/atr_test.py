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

# set focus
xInput = 20
yInput = 0
zInput = 80

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system


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
execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

arr.sort_by_pin()
kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
karray = get_karray(arr)
medium = get_medium(params, ref_values_only=ref_values_only)
sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
source = get_source(kgrid, karray, source_mat)

output = kspaceFirstOrder3D(kgrid=kgrid,
                                source=source,
                                sensor=sensor,
                                medium=medium,
                                simulation_options=simulation_options,
                                execution_options=execution_options)

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


# # delay_method = delay_methods.TimeReversal()
# # apod_method = apod_methods.MaxAngle()
# # apod_method = apod_methods.PiecewiseLinear()

# sim_setup = SimSetup(x_extent=(-55,55), y_extent=(-30,30), z_extent=(-4,150))
# protocol = Protocol(
#     id='test_protocol',
#     name='Test Protocol',
#     pulse=pulse,
#     sequence=sequence,
#     focal_pattern=focal_pattern,
#     apod_method=apod_method,
#     sim_setup=sim_setup)

# solution, sim_res, scaled_analysis = protocol.calc_solution(
#     target=target,
#     transducer=arr,
#     simulate=True,
#     scale=True,
#     use_gpu=True)
# voltage = solution.voltage
# peak_to_peak_voltage = solution.voltage * 2 # Peak to peak voltage for the pulse

# logger.info(f'Apodizations: {solution.apodizations}')
# logger.info(f'Delays: {solution.delays}')

# profile_index = 1
# profile_increment = True
# trigger_mode = "continuous"

# duty_cycle = int((duration_msec/interval_msec) * 100)
# if duty_cycle > 50:
#     logger.warning("❗❗ Duty cycle is above 50% ❗❗")

# p_map = sim_res['p_max']
# if scaled_analysis is not None:
#     scaled_analysis.to_table().set_index('Param')[['Value', 'Units', 'Status']]

# plt.figure()
# plt.imshow(p_map[:,30,:])
# plt.colorbar()
# plt.figure()
# plt.show()
