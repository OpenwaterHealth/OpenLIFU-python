from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from serial.serialutil import SerialException

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
test_id = "burn_in_test"
test_name = "Burn-In Test"
HW_SIMULATE = False # HW simulation flag

def format_hhmmss(seconds):
    """Format seconds into HH:MM:SS string."""
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if days > 0:
        return f"{int(days)}d {int(hours):02}h{int(minutes):02}m{int(secs):02}s"
    elif hours > 0:
        return f"{int(hours)}h{int(minutes):02}m{int(secs):02}s"
    elif minutes > 0:
        return f"{int(minutes)}m{int(secs):02}s"
    else:
        return f"{secs:0.1f}s"

# ------------------- Test Case Definitions -------------------
frequencies_kHz = {1: 150, 2: 400}

test_cases = {
    0: {
        'id': 'dryrun',
        'description': 'Dry Run Testing',
        'voltage': 60,
        'duty_cycle_pct': 5,
        'sequence_duration_sec': 3,
        'sequence_repeats': 3,
        'sequence_repeat_interval_sec': 6,
        'test_repeats': 2,
        'test_repeat_interval_sec': 30
    },
    1: {
        'id': '24hr',
        'description': '24 hour burn-in',
        'voltage': 60,
        'duty_cycle_pct': 5,
        'sequence_duration_sec': 10*60,
        'sequence_repeats': 8,
        'sequence_repeat_interval_sec': 60*60,
        'test_repeats': 1,
        'test_repeat_interval_sec': 60*60*24
    },
    2: {
        'id': 'lifetime',
        'description': 'Lifetime test',
        'voltage': 60,
        'duty_cycle_pct': 5,
        'sequence_duration_sec': 10*60,
        'sequence_repeats': 8,
        'sequence_repeat_interval_sec': 60*60,
        'test_repeats': 200,
        'test_repeat_interval_sec': 60*60*24
    }
}


pulse_interval_msec = 200
num_modules = 2
use_external_power_supply = False
CONSOLE_SHUTOFF_TEMP_C = 70.0
INITIAL_TEMP_SHUTOFF_C = 40
INITIAL_TIME_WINDOW = 5
RAPID_TEMP_INCREASE_PER_SECOND_SHUTOFF_C = 3
TEMPERATURE_CHECK_INTERVAL = 1
TEMPERATURE_LOG_INTERVAL = 5
TIME_LOG_INTERVAL = 1
TIME_CHECK_INTERVAL = 0.1

# ------------------- User Input Section -------------------
print("Select Test Case:")
for i, case in test_cases.items():
    print(f"{i}. {case['description']}, {case['voltage']}V, {case['duty_cycle_pct']}%, {format_hhmmss(case['sequence_duration_sec'])} sequence, repeated {case['sequence_repeats']}x every {format_hhmmss(case['sequence_repeat_interval_sec'])}, test repeated {case['test_repeats']}x every {format_hhmmss(case['test_repeat_interval_sec'])}")
while True:
    choice = input(f"Select test case by number {list(test_cases.keys())}: ").strip()
    if choice.isdigit() and int(choice) in test_cases:
        selected_case = test_cases[int(choice)]
        voltage = selected_case['voltage']
        duty_cycle_pct = selected_case['duty_cycle_pct']
        sequence_duration_sec = selected_case['sequence_duration_sec']
        sequence_repeats = selected_case['sequence_repeats']
        sequence_repeat_interval_sec = selected_case['sequence_repeat_interval_sec']
        test_repeats = selected_case['test_repeats']
        test_repeat_interval_sec = selected_case['test_repeat_interval_sec']
        print(f"Selected Test Case: {selected_case['description']}")
        break
    print("Invalid selection. Please try again.")


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


total_test_time = test_repeat_interval_sec * (test_repeats-1) + (sequence_repeats-1)*sequence_repeat_interval_sec + sequence_duration_sec
duration_msec = int(duty_cycle_pct/100 * pulse_interval_msec)

test_case_description = selected_case["description"]
test_case_long_description = f"{frequency_kHz}kHz, {voltage}V, {duty_cycle_pct}%, {format_hhmmss(sequence_duration_sec)} sequence, repeated {sequence_repeats}x every {format_hhmmss(sequence_repeat_interval_sec)}, test repeated {test_repeats}x every {format_hhmmss(test_repeat_interval_sec)}"
test_case_id = f"{frequency_kHz}kHz_{selected_case['id']}"


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
logger.info(f"Total test time if all sequences complete: {format_hhmmss(total_test_time)}")

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
shutdown_event = threading.Event()
sequence_complete_event = threading.Event()
temperature_shutdown_event = threading.Event()

def monitor_temperature(
        initial_temp_shutoff_C=INITIAL_TEMP_SHUTOFF_C,
        initial_time_window=INITIAL_TIME_WINDOW,
        rapid_temp_increase_per_second_shutoff_C=RAPID_TEMP_INCREASE_PER_SECOND_SHUTOFF_C,
        console_shutoff_temp_C=CONSOLE_SHUTOFF_TEMP_C,
        temperature_log_interval=TEMPERATURE_LOG_INTERVAL,
        temperature_check_interval=TEMPERATURE_CHECK_INTERVAL,
        max_serial_failures=3):
    serial_failures = 0
    start_time = time.time()
    last_log_time = 0
    prev_tx_temp = None
    prev_amb_temp = None
    prev_con_temp = None
    while True:
        if shutdown_event.is_set():
            return

        time_elapsed = time.time() - start_time
        # Check temperatures
        try:
            if not use_external_power_supply:
                if prev_con_temp is None:
                    prev_con_temp = interface.hvcontroller.get_temperature1()
                con_temp = interface.hvcontroller.get_temperature1()
            if prev_tx_temp is None:
                prev_tx_temp = interface.txdevice.get_temperature()
            tx_temp = interface.txdevice.get_temperature()
            if prev_amb_temp is None:
                prev_amb_temp = interface.txdevice.get_ambient_temperature()
            amb_temp = interface.txdevice.get_ambient_temperature()
        except SerialException as e:
            logger.error(f"SerialException encountered while reading temperatures: {e}")
            serial_failures += 1
            if serial_failures >= max_serial_failures:
                logger.critical("Maximum serial failures reached. Initiating shutdown.")
                break
            time.sleep(temperature_check_interval)
            continue

        # Log Temperature every temperature_log_interval seconds
        time_since_last_log = time_elapsed - last_log_time
        if time_since_last_log >= temperature_log_interval:
            last_log_time = time_elapsed
            if not use_external_power_supply:
                logger.info(f"  Console Temp: {con_temp}°C, TX Temp: {tx_temp}°C, Ambient Temp: {amb_temp}°C")
            else:
                logger.info(f"TX Temp: {tx_temp}°C, Ambient Temp: {amb_temp}°C")

        # Check for initial temperature rise at startup
        within_initial_time_threshold = time_elapsed < initial_time_window
        if within_initial_time_threshold:
            if not use_external_power_supply and (con_temp > initial_temp_shutoff_C):
                logger.warning(f"Console temperature {con_temp}°C exceeds initial shutoff threshold of {initial_temp_shutoff_C}°C within {initial_time_window}s.")
                break
            if (tx_temp > initial_temp_shutoff_C):
                logger.warning(f"TX device temperature {tx_temp}°C exceeds initial shutoff threshold of {initial_temp_shutoff_C}°C within {initial_time_window}s.")
                break
            if (amb_temp > initial_temp_shutoff_C):
                logger.warning(f"Ambient temperature {amb_temp}°C exceeds initial shutoff threshold of {initial_temp_shutoff_C}°C within {initial_time_window}s.")
                break

        # Check for rapid temperature increase
        if not use_external_power_supply:
            if (con_temp - prev_con_temp) > rapid_temp_increase_per_second_shutoff_C:
                logger.warning(f"Console temperature rose from {prev_con_temp}°C to {con_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
                break
            prev_con_temp = con_temp
        if (tx_temp - prev_tx_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"TX device temperature rose from {prev_tx_temp}°C to {tx_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
            break
        prev_tx_temp = tx_temp
        if (amb_temp - prev_amb_temp) > rapid_temp_increase_per_second_shutoff_C:
            logger.warning(f"Ambient temperature rose from {prev_amb_temp}°C to {amb_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {temperature_check_interval}s.")
            break
        prev_amb_temp = amb_temp

        if not use_external_power_supply and (con_temp > console_shutoff_temp_C):
            logger.warning(f"Console temperature {con_temp}°C exceeds shutoff threshold {console_shutoff_temp_C}°C.")
            break
        if tx_temp > 70.0:
            logger.warning(f"TX device temperature {tx_temp}°C exceeds shutoff threshold 70.0°C.")
            break
        if amb_temp > 70.0:
            logger.warning(f"Ambient temperature {amb_temp}°C exceeds shutoff threshold 70.0°C.")
            break
        time.sleep(temperature_check_interval)

    logger.warning("Temperature shutdown triggered.")
    shutdown_event.set()
    temperature_shutdown_event.set()

def exit_on_time_complete(total_test_time,
                          time_log_interval=TIME_LOG_INTERVAL,
                          check_interval=TIME_CHECK_INTERVAL):
    start = time.time()
    elapsed_time = 0
    last_log_time = 0
    while True:
        if shutdown_event.is_set():
            return
        time.sleep(check_interval)
        elapsed_time = time.time() - start
        time_since_last_log = elapsed_time - last_log_time
        if time_since_last_log >= time_log_interval:
            last_log_time = elapsed_time
            remaining_time = total_test_time - elapsed_time
            if remaining_time < 0:
                remaining_time = 0
            logger.info(f"  Sequence time elapsed: {format_hhmmss(elapsed_time)}, Remaining time: {format_hhmmss(remaining_time)}")
        if elapsed_time >= total_test_time:
            logger.info(f"  ✅ Sequence complete {format_hhmmss(total_test_time)} reached.")
            shutdown_event.set()
            sequence_complete_event.set()
            return

# ------------------- Solution Setup -------------------
pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=pulse_interval_msec*1e-3,
    pulse_count=int(sequence_duration_sec/(pulse_interval_msec*1e-3)),
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

start_time = time.time()
test_number = 1
try:
    while test_number <= test_repeats:
        logger.info(f"[{format_hhmmss(time.time()-start_time)}]Starting Test {test_number}/{test_repeats}...")
        test_start_time = time.time()
        sequence_number = 1
        while sequence_number <= sequence_repeats:
            logger.info(f"[{format_hhmmss(time.time()-start_time)}]  Starting Sequence {sequence_number}/{sequence_repeats}...")
            sequence_complete_event.clear()
            sequence_start_time = time.time()
            # ------------------- Start Test -------------------
            temp_thread = threading.Thread(target=monitor_temperature)
            completion_thread = threading.Thread(target=exit_on_time_complete, args=(sequence_duration_sec,))
            all_threads = [temp_thread, completion_thread]
            if interface.start_sonication():
                for t in all_threads:
                    t.start()
                while all(t.is_alive() for t in all_threads):
                    time.sleep(0.1)
                for t in all_threads:
                    t.join()
                if not interface.stop_sonication():
                    logger.error("Failed to stop trigger.")
                    break
            else:
                logger.error("Failed to start trigger.")
            if sequence_complete_event.is_set():
                logger.info(f"[{format_hhmmss(time.time()-start_time)}] Test ({test_number}/{test_repeats}), Sequence ({sequence_number}/{sequence_repeats}) complete")
                if sequence_number < sequence_repeats:
                    off_time = sequence_repeat_interval_sec - (time.time() - sequence_start_time)
                    if off_time > 0:
                        logger.info(f"  Waiting {format_hhmmss(off_time)} until next sequence...")
                        time.sleep(off_time)
                sequence_number += 1
                shutdown_event.clear()
            else:
                break
        if sequence_complete_event.is_set():
            logger.info(f"[{format_hhmmss(time.time()-start_time)}] Test {test_number}/{test_repeats} complete")
            if test_number < test_repeats:
                off_time = test_repeat_interval_sec - (time.time() - test_start_time)
                if off_time > 0:
                    logger.info(f"Waiting {format_hhmmss(off_time)} until next test...")
                    time.sleep(off_time)
            test_number += 1
            shutdown_event.clear()
        else:
            break
except KeyboardInterrupt:
    logger.warning("Test aborted by user KeyboardInterrupt.")
    shutdown_event.set()
    test_status = "aborted by user"
finally:
    if not interface.stop_sonication():
        logger.critical("Failed to stop trigger.")
        test_status = "error"
    elif temperature_shutdown_event.is_set():
        test_status = "temperature shutdown"
    elif sequence_complete_event.is_set() and (test_number > test_repeats):
        test_status = "passed"
    else:
        test_status = "error"


logger.info(f"[{format_hhmmss(time.time()-start_time)}] All tests complete or aborted. Cleaning up...")

if not use_external_power_supply:
    logger.info("Turning off HV and 12V...")
    interface.hvcontroller.turn_hv_off()
    interface.hvcontroller.turn_12v_off()
    logger.info("HV and 12V turned off.")

for t in all_threads:
    t.join()

if test_status == "passed":
    logger.info(f"✅ TEST PASSED: {test_case_description} completed successfully.")
elif test_status == "temperature shutdown":
    logger.info(f"❌ TEST FAILED: {test_case_description} failed due to temperature shutdown.")
elif test_status == "aborted by user":
    logger.info(f"❌ TEST ABORTED: {test_case_description} aborted by user.")
else:
    logger.info(f"❌ TEST FAILED: {test_case_description} failed due to unexpected error.")
