from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if os.name == 'nt':
    pass
else:
    pass

import numpy as np

import openlifu
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
openlifu_dir = Path(openlifu.__file__).parent.parent.parent.resolve()
test_id = "thermal_stress_test"
test_name = "Thermal Stress Test"
HW_SIMULATE = False

# ------------------- Test Case Definitions -------------------
test_cases = {
    0: {
        'id': 'dryrun',
        'description': 'Dry Run Testing',
        'voltage': 30,
        'duty_cycle_pct': 5,
        'sequence_duration': 10  # seconds
    },
    1: {
        'id': 'hv',
        'description': 'High Voltage, Low Duty Cycle, 10min',
        'voltage': 60,
        'duty_cycle_pct': 5,
        'sequence_duration': 10 * 60  # seconds
    },
    2: {
        'id': 'med',
        'description': 'Medium Voltage, High Duty Cycle, 2min',
        'voltage': 30,
        'duty_cycle_pct': 50,
        'sequence_duration': 2 * 60  # seconds
    },
    3: {
        'id': 'lv',
        'description': 'Low Voltage, High Duty Cycle, 10min',
        'voltage': 20,
        'duty_cycle_pct': 50,
        'sequence_duration': 10 * 60  # seconds
    },
}

frequencies_kHz = {1: 150, 2: 400}

interval_msec = 200
num_modules = 2
use_external_power_supply = False
console_shutoff_temp_C = 70.0
rapid_temp_shutoff_C = 40
rapid_temp_shutoff_seconds = 5
rapid_temp_increase_per_second_shutoff_C = 3
temperature_check_interval = 1

# ------------------- User Input Section -------------------
print("Choose Frequency:")
for i, freq in frequencies_kHz.items():
    print(f"{i}. {freq} kHz")

while True:
    freq_choice = input(f"Select frequency by number {list(frequencies_kHz.keys())}: ").strip()
    if freq_choice.isdigit() and int(freq_choice) in frequencies_kHz:
        frequency_kHz = frequencies_kHz[int(freq_choice)]
        print(f"Selected Frequency: {frequency_kHz} kHz")
        break
    print("Invalid selection. Please try again.")

def format_hhmmss(seconds):
    """Format seconds into HH:MM:SS string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{int(hours)}:{int(minutes):02}:{int(secs):02}"
    else:
        return f"{int(minutes):02}:{int(secs):02}"

print("\nAvailable Burn-in Test Cases:")
for i, case in test_cases.items():
    print(f"{i}. {case['voltage']}V, {case['duty_cycle_pct']}% Duty Cycle, {format_hhmmss(case['sequence_duration'])} total")

while True:
    choice = input(f"Select a test case by number {list(test_cases.keys())}: ").strip()
    if choice.isdigit() and int(choice) in test_cases:
        test_case_num = int(choice)
        test_case = test_cases[test_case_num]
        print(f"Selected Test Case: {test_case_num}\n")
        break
    print("Invalid selection. Please try again.")

test_case_description = test_case["description"]
test_case_long_description = f"{frequency_kHz}kHz, Case {test_case_num}: {test_case['voltage']}V, {test_case['duty_cycle_pct']}%, {format_hhmmss(test_case['sequence_duration'])}"
test_case_id = f"{frequency_kHz}kHz_{test_case['id']}"
voltage = test_case['voltage']
duration_msec = int(test_case['duty_cycle_pct']/100 * interval_msec)
sequence_duration = test_case['sequence_duration']


# ------------------- Logging Setup -------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = openlifu_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handle = logging.FileHandler(log_dir / f"{test_id}_{test_case_id}_{timestamp}.log", mode='w', encoding='utf-8')
    logger.addHandler(file_handle)


if HW_SIMULATE:
    logger.info(f"Beginning Test {test_case_description} (TEST MODE)")
    logger.info("⚠️ TEST MODE: This is a simulated test run.")
    logger.info("No actual hardware interactions will occur.")
else:
    logger.info(f"Beginning Test {test_case_description}")
logger.info(f"{test_case_long_description}")


# ------------------- Device Setup -------------------

db_path = openlifu_dir / "db_dvc"
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

logger.info(f"Starting {test_name}...")
interface = LIFUInterface(ext_power_supply=use_external_power_supply, TX_test_mode=HW_SIMULATE, HV_test_mode=HW_SIMULATE)
tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply and not tx_connected:
    logger.warning("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()
    time.sleep(2)
    interface.stop_monitoring()
    del interface
    time.sleep(1)
    logger.info("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface(ext_power_supply=use_external_power_supply, TX_test_mode=HW_SIMULATE, HV_test_mode=HW_SIMULATE)
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
test_status = "not started"
shutdown_event = threading.Event()
sequence_complete_event = threading.Event()
temperature_shutdown_event = threading.Event()

temperature_log_interval = 5  # seconds
def monitor_temperature():
    start = time.time()
    time_since_last_log = 0
    temperature_shutdown = False
    prev_tx_temp = None
    prev_amb_temp = None
    prev_con_temp = None
    while not (shutdown_event.is_set() or temperature_shutdown):
        elapsed = time.time() - start
        within_initial_time_threshold = elapsed < rapid_temp_shutoff_seconds
        if not use_external_power_supply:
            if prev_con_temp is None:
                prev_con_temp = interface.hvcontroller.get_temperature1()
            con_temp = interface.hvcontroller.get_temperature1()
            if (con_temp - prev_con_temp) > rapid_temp_increase_per_second_shutoff_C:
                logger.warning(f"Console temperature rose from {prev_con_temp}°C to {con_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
                temperature_shutdown = True
            else:
                prev_con_temp = con_temp
        if prev_tx_temp is None:
            prev_tx_temp = interface.txdevice.get_temperature()
        tx_temp = interface.txdevice.get_temperature()
        if (tx_temp - prev_tx_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"TX device temperature rose from {prev_tx_temp}°C to {tx_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
            temperature_shutdown = True
        else:
            prev_tx_temp = tx_temp
        if prev_amb_temp is None:
            prev_amb_temp = interface.txdevice.get_ambient_temperature()
        amb_temp = interface.txdevice.get_ambient_temperature()
        if (amb_temp - prev_amb_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"Ambient temperature rose from {prev_amb_temp}°C to {amb_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
            temperature_shutdown = True
        else:
            prev_amb_temp = amb_temp
        time_since_last_log += temperature_check_interval
        if time_since_last_log >= temperature_log_interval:
            time_since_last_log = 0
            if not use_external_power_supply:
                logger.info(f"Console Temp: {con_temp}°C, TX Temp: {tx_temp}°C, Ambient Temp: {amb_temp}°C")
            else:
                logger.info(f"TX Temp: {tx_temp}°C, Ambient Temp: {amb_temp}°C")
        if within_initial_time_threshold:
            if not use_external_power_supply and (con_temp > rapid_temp_shutoff_C):
                logger.warning(f"Console temperature {con_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                temperature_shutdown = True
            elif (tx_temp > rapid_temp_shutoff_C):
                logger.warning(f"TX device temperature {tx_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                temperature_shutdown = True
            elif (amb_temp > rapid_temp_shutoff_C):
                logger.warning(f"Ambient temperature {amb_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                temperature_shutdown = True
        if not use_external_power_supply and (con_temp > console_shutoff_temp_C):
            logger.warning(f"Console temperature {con_temp}°C exceeds shutoff threshold {console_shutoff_temp_C}°C.")
            temperature_shutdown = True
        if tx_temp > 70.0:
            logger.warning(f"TX device temperature {tx_temp}°C exceeds shutoff threshold 70.0°C.")
            temperature_shutdown = True
        if amb_temp > 70.0:
            logger.warning(f"Ambient temperature {amb_temp}°C exceeds shutoff threshold 70.0°C.")
            temperature_shutdown = True
        time.sleep(temperature_check_interval)
    if not shutdown_event.is_set():
        shutdown_event.set()
    if temperature_shutdown:
        logger.warning("Temperature shutdown triggered.")
        temperature_shutdown_event.set()

def exit_on_time_complete(timeout, time_log_interval=2, check_interval=0.1):
    start = time.time()
    elapsed_time = 0
    last_log_time = 0
    while not (shutdown_event.is_set() or (elapsed_time >= timeout)):
        time.sleep(check_interval)
        elapsed_time = time.time() - start
        time_since_last_log = elapsed_time - last_log_time
        if time_since_last_log >= time_log_interval:
            last_log_time = elapsed_time
            remaining_time = timeout - elapsed_time
            if remaining_time < 0:
                remaining_time = 0
            logger.info(f"Time elapsed: {format_hhmmss(elapsed_time)}, Remaining time: {format_hhmmss(remaining_time)}")
    if not shutdown_event.is_set():
        logger.info(f"✅ Test complete {format_hhmmss(timeout)} reached.")
        shutdown_event.set()
        sequence_complete_event.set()
    else:
        logger.warning(f"❌ Test shutdown early due to event at {format_hhmmss(elapsed_time)}.")



# ------------------- Solution Setup -------------------
pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(sequence_duration/(interval_msec*1e-3)),
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

logger.info(f"Press enter to START {test_case_description}: ")
input()

# ------------------- Start Test -------------------
temp_thread = threading.Thread(target=monitor_temperature)
completion_thread = threading.Thread(target=exit_on_time_complete, args=(sequence_duration,))
all_threads = [temp_thread, completion_thread]

logger.info("Starting Trigger...")
if interface.start_sonication():
    logger.info("Trigger Running... (Press CTRL-C to stop early)")

    for t in all_threads:
        t.start()
    try:
        while all(t.is_alive() for t in all_threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.warning("Test aborted by user KeyboardInterrupt.")
        test_status = "aborted by user"
        shutdown_event.set()
    finally:
        if not shutdown_event.is_set():
            logger.warning("A thread exited without setting shutdown event.")
            shutdown_event.set()
        if interface.stop_sonication():
            logger.info("Trigger stopped successfully.")
        else:
            logger.error("Failed to stop trigger.")
        for t in all_threads:
            t.join()

else:
    logger.error("Failed to start trigger.")
    test_status = "error"

if sequence_complete_event.is_set():
    test_status = "passed"
elif temperature_shutdown_event.is_set():
    test_status = "temperature shutdown"

if not use_external_power_supply:
    logger.info("Turning off HV and 12V...")
    interface.hvcontroller.turn_hv_off()
    interface.hvcontroller.turn_12v_off()
    logger.info("HV and 12V turned off.")

if test_status == "passed":
    logger.info(f"✅ TEST PASSED: {test_case_description} completed successfully.")
elif test_status == "temperature shutdown":
    logger.info(f"❌TEST FAILED: {test_case_description} failed due to temperature shutdown.")
elif test_status == "aborted by user":
    logger.info(f"❌TEST ABORTED: {test_case_description} aborted by user.")
else:
    logger.info(f"❌TEST FAILED: {test_case_description} failed due to unexpected error.")
