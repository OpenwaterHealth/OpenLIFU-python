from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import select

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

"""
Burn-in Test Script
- User selects one of three test cases.
- Test runs for a fixed total duration or until a thermal shutdown occurs.
- Logs temperature and device status.
"""

one_day_in_seconds = 24 * 60 * 60

# ------------------- Logging Setup -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handle = logging.FileHandler(f"burn_in_test_{timestamp}.log", mode='w', encoding='utf-8')
    logger.addHandler(file_handle)

# ------------------- Test Case Definitions -------------------
test_cases = [
    {
        'name': 'Case 1: 65V, 10ms pulse, 10 minutes',
        'voltage': 65,
        'duration_msec': 10,
        'total_test_time': 10 * 60,  # seconds
    },
    {
        'name': 'Case 2: 30V, 100ms pulse, 2 minutes',
        'voltage': 30,
        'duration_msec': 100,
        'total_test_time': 2 * 60,  # seconds
    },
    {
        'name': 'Case 3: 20V, 100ms pulse, 10 minutes',
        'voltage': 20,
        'duration_msec': 100,
        'total_test_time': 10 * 60,  # seconds
    },
]

test_duration = [
    {
        'name': 'Case 1: Extended Test, 1 day',
        'run_time': one_day_in_seconds,
        'times_per_day': 8,
    },
    {
        'name': 'Case 2: Lifetime Test, 365 days',
        'run_time': 365*one_day_in_seconds,
        'times_per_day': 8,
    },
]


frequency_kHz = 400
interval_msec = 200
num_modules = 2
use_external_power_supply = False
console_shutoff_temp_C = 70.0
rapid_temp_shutoff_C = 40
rapid_temp_shutoff_seconds = 5
rapid_temp_increase_per_second_shutoff_C = 3
log_interval = 1

# ------------------- User Input Section -------------------
logger.info("Available Burn-in Test Cases:")
for case in test_cases:
    logger.info(f"{case['name']}")

while True:
    choice = input("\nSelect a test case by number (1, 2, 3): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(test_cases):
        selected = test_cases[int(choice)-1]
        break
    logger.info("Invalid selection. Please enter 1, 2, or 3.")

voltage = selected['voltage']
duration_msec = selected['duration_msec']
total_test_time = selected['total_test_time']

logger.info(f"Case {choice} Selected\n\n")

logger.info("Select a Test Duration (1, 2):")
for case in test_duration:
    logger.info(f"{case['name']} {case['times_per_day']} times per day for {case['run_time']//(one_day_in_seconds)} day{'s' if case['run_time']//(one_day_in_seconds) > 1 else ''}")

while True:
    choice = input("\nSelect a test case by number (1, 2): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(test_duration):
        selected = test_duration[int(choice)-1]
        break
    logger.info("Invalid selection. Please enter 1 or 2.")

run_time = selected['run_time']
times_per_day = selected['times_per_day']

logger.info(f"Case {choice} Selected\n")
logger.info(f"This will run the burn-in test {times_per_day} times a day for {run_time//(one_day_in_seconds)} day{'s' if run_time//(one_day_in_seconds) > 1 else ''}.\n")

# ------------------- Device Setup -------------------
here = Path(__file__).parent.resolve()
db_path = here / ".." / "db_dvc"
db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1")
arr.sort_by_pin()

xInput, yInput, zInput = 0, 0, 50
target = Point(position=(xInput, yInput, zInput), units="mm")
focus = target.get_position(units="mm")
distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1)).reshape(1, -1)
tof = distances * 1e-3 / 1500
delays = tof.max() - tof
apodizations = np.ones((1, arr.numelements()))

logger.info("Starting Burn-in Test...")
interface = LIFUInterface(ext_power_supply=use_external_power_supply)
tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply and not tx_connected:
    logger.warning("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()
    time.sleep(2)
    interface.stop_monitoring()
    del interface
    time.sleep(1)
    logger.info("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface(ext_power_supply=use_external_power_supply)
    tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply:
    if hv_connected:
        logger.info(f"  HV Connected: {hv_connected}")
    else:
        logger.error("❌ HV NOT fully connected.")
        sys.exit(1)
else:
    logger.info("  Using external power supply")

if tx_connected:
    logger.info(f"  TX Connected: {tx_connected}")
    logger.info("✅ LIFU Device fully connected.")
else:
    logger.error("❌ TX NOT fully connected.")
    sys.exit(1)

# ------------------- Temperature Monitoring -------------------
stop_logging = False
shutdown_event = threading.Event()

def monitor_temperature():
    start = time.time()
    shutdown = False
    prev_tx_temp = None
    prev_amb_temp = None
    prev_con_temp = None
    while not (stop_logging or shutdown):
        elapsed = time.time() - start
        if elapsed > total_test_time:
            logger.info(f"Total test time {total_test_time}s reached. Stopping test.")
            shutdown = True
        within_initial_time_threshold = elapsed < rapid_temp_shutoff_seconds
        if not use_external_power_supply:
            if prev_con_temp is None:
                prev_con_temp = interface.hvcontroller.get_temperature1()
                prev_con_temp = 0
            con_temp = interface.hvcontroller.get_temperature1()
            con_temp = 0
            if (con_temp - prev_con_temp) > rapid_temp_increase_per_second_shutoff_C:
                logger.warning(f"Console temperature rose from {prev_con_temp}°C to {con_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                shutdown = True
            else:
                prev_con_temp = con_temp
        if prev_tx_temp is None:
            prev_tx_temp = interface.txdevice.get_temperature()
        tx_temp = interface.txdevice.get_temperature()
        if (tx_temp - prev_tx_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"TX device temperature rose from {prev_tx_temp}°C to {tx_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
            shutdown = True
        else:
            prev_tx_temp = tx_temp
        if prev_amb_temp is None:
            prev_amb_temp = interface.txdevice.get_ambient_temperature()
        amb_temp = interface.txdevice.get_ambient_temperature()
        if (amb_temp - prev_amb_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"Ambient temperature rose from {prev_amb_temp}°C to {amb_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
            shutdown = True
        else:
            prev_amb_temp = amb_temp
        if within_initial_time_threshold:
            if not use_external_power_supply and (con_temp > rapid_temp_shutoff_C):
                logger.warning(f"Console temperature {con_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                shutdown = True
            elif (tx_temp > rapid_temp_shutoff_C):
                logger.warning(f"TX device temperature {tx_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                shutdown = True
            elif (amb_temp > rapid_temp_shutoff_C):
                logger.warning(f"Ambient temperature {amb_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                shutdown = True
        if not use_external_power_supply and (con_temp > console_shutoff_temp_C):
            logger.warning(f"Console temperature {con_temp}°C exceeds shutoff threshold {console_shutoff_temp_C}°C.")
            shutdown = True
        if tx_temp > 70.0:
            logger.warning(f"TX device temperature {tx_temp}°C exceeds shutoff threshold 70.0°C.")
            shutdown = True
        if amb_temp > 70.0:
            logger.warning(f"Ambient temperature {amb_temp}°C exceeds shutoff threshold 70.0°C.")
            shutdown = True
        if shutdown:
            interface.txdevice.stop_trigger()
            shutdown_event.set()
            break
        time.sleep(log_interval)
    logger.info(f"Temperature monitoring stopped after {int(elapsed//60)}:{int(elapsed%60):02d}.")

# ------------------- Solution Setup -------------------
pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(total_test_time/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=1
)
pin_order = np.argsort([el.pin for el in arr.elements])
solution = Solution(
    delays=delays[:, pin_order],
    apodizations=apodizations[:, pin_order],
    transducer=arr,
    pulse=pulse,
    voltage=voltage,
    sequence=sequence
)
profile_index = 1
profile_increment = True
trigger_mode = "continuous"

interface.set_solution(
    solution=solution,
    profile_index=profile_index,
    profile_increment=profile_increment,
    trigger_mode=trigger_mode
)

logger.info("Press enter to START burn-in test with the following parameters: ")
logger.info(f"  Voltage: {voltage}V")
logger.info(f"  Pulse Duration: {duration_msec}ms")
logger.info(f"  Duration: {total_test_time//60} minutes")
logger.info(f"  Times per day: {times_per_day}")
logger.info(f"  Total Run Time: {run_time//(one_day_in_seconds)} day{'s' if run_time//(one_day_in_seconds) > 1 else ''}.")
input()

# ------------------- Start Test -------------------
temp_thread = threading.Thread(target=monitor_temperature)

# if test duration is 10 mins, and need to run 8 times in a 24 hour period, so start every 170 mins
# initiating test 1 of 8
# waiting 170 mins until next test

time_between_tests = ((one_day_in_seconds - (total_test_time * times_per_day))/(60*times_per_day))
logger.info(f"Time between tests: {time_between_tests} minutes")

for i in range(run_time//one_day_in_seconds):
    for j in range(times_per_day):
        logger.info(f"Initiating test {j+1} of {times_per_day} for day {i+1}.")

        logger.info("Starting Trigger...")
        if interface.start_sonication():
            logger.info("Trigger Running... (Press enter to STOP early)")
            def input_wrapper():
                if os.name == 'nt':
                    while not shutdown_event.is_set():
                        if msvcrt.kbhit():
                            char = msvcrt.getch()
                            if char == b'\r':
                                shutdown_event.set()
                                break
                        time.sleep(0.1)
                else:
                    while not shutdown_event.is_set():
                        ready, _, _ = select.select([], [], [], 0.1)
                        if ready:
                            sys.stdin.readline()
                            shutdown_event.set()
                            break
            user_input = threading.Thread(target=input_wrapper)

            temp_thread.start()
            user_input.start()
            while (temp_thread.is_alive() and user_input.is_alive()):
                time.sleep(0.1)
            shutdown_event.set()
            stop_logging = True

            temp_thread.join()
            user_input.join()
            if interface.stop_sonication():
                logger.info("Trigger stopped successfully.")
            else:
                logger.error("Failed to stop trigger.")
        else:
            logger.error("Failed to start trigger.")

        logger.info(f"Burn-in test {j+1} of {times_per_day} for day {i+1} complete. Shutting down devices if needed.")
        if not use_external_power_supply:
            interface.hvcontroller.turn_hv_off()
            interface.hvcontroller.turn_12v_off()

        logger.info(f"Waiting {int(time_between_tests)} minutes until next test.")
