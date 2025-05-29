from __future__ import annotations

import logging
import os
import sys
import threading
import time

if os.name == 'nt':
    import msvcrt
else:
    import select

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_thermal_stress.py

"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

log_interval = 1  # seconds; you can adjust this variable as needed

frequency_kHz = 400 # Frequency in kHz
voltage = 50.0 # Voltage in Volts
duration_msec = 5 # Pulse Duration in milliseconds
interval_msec = 100 # Pulse Repetition Interval in milliseconds
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
rapid_temp_increase_per_second_shutoff_C = 2 # Rapid temperature climbing shutoff in Celsius

peak_to_peak_voltage = voltage * 2 # Peak to peak voltage for the pulse

logger.info("Starting LIFU Test Script...")
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply and not tx_connected:
    logger.warning("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()

    # Give time for the TX device to power up and enumerate over USB
    time.sleep(2)

    # Cleanup and recreate interface to reinitialize USB devices
    interface.stop_monitoring()
    del interface
    time.sleep(1)  # Short delay before recreating

    logger.info("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface()

    # Re-check connection
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

stop_logging = False  # flag to signal the logging thread to stop

def log_temperature():
    # Create a file with the current timestamp in the name
    start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{frequency_kHz}kHz_{voltage}V_{duration_msec}ms_Duration_{interval_msec}ms_Interval_Temperature_Readings.csv"
    shutdown = False

    prev_tx_temp = None
    prev_amb_temp = None
    prev_con_temp = None

    if use_external_power_supply:
        con_temp = "N/A"

    with open(filename, "w") as logfile:
        # Create header for CSV file
        log_line = "Current Time,Frequency (kHz),Duration (ms),Interval (ms),Voltage (Per Rail),Voltage (Peak to Peak),Console Temperature (°C),Transmitter Temperature (°C),Ambient Temperature (°C)\n"
        logfile.write(log_line)
        logfile.flush()  # Ensure the data is written immediately
        while not (stop_logging or shutdown):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # Timer for initial rapid temperature increase
            within_initial_time_threshold = (time.time() - start) < rapid_temp_shutoff_seconds

            ## Check for too high of a temperature increase between readings
            # Console general temperature increase (bypass if using external power supply)
            if not use_external_power_supply:
                if prev_con_temp is None:
                    prev_con_temp = interface.hvcontroller.get_temperature1()
                con_temp = interface.hvcontroller.get_temperature1()
                if (con_temp - prev_con_temp) > rapid_temp_increase_per_second_shutoff_C:
                    logger.warning(f"Console temperature rose from {prev_con_temp}°C to {con_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                    log_line = f"{current_time},SHUTDOWN,Console temperature exceeded rapid temp increase shutoff threshold\n"
                    shutdown=True
                else:
                    prev_con_temp = con_temp

            # TX device general temperature increase
            if prev_tx_temp is None:
                prev_tx_temp = interface.txdevice.get_temperature()
            tx_temp = interface.txdevice.get_temperature()
            if (tx_temp - prev_tx_temp) > rapid_temp_increase_per_second_shutoff_C:
                logger.warning(f"TX device temperature rose from {prev_tx_temp}°C to {tx_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded rapid temp increase shutoff threshold\n"
                shutdown=True
            else:
                prev_tx_temp = tx_temp

            # Ambient temperature general increase
            if prev_amb_temp is None:
                prev_amb_temp = interface.txdevice.get_ambient_temperature()
            amb_temp = interface.txdevice.get_ambient_temperature()
            if (amb_temp - prev_amb_temp) > rapid_temp_increase_per_second_shutoff_C:
                logger.warning(f"Ambient temperature rose from {prev_amb_temp}°C to {amb_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded rapid temp increase shutoff threshold\n"
                shutdown=True
            else:
                prev_amb_temp = amb_temp

            # Check for initial rapid temperature increase
            if (within_initial_time_threshold):
                if not use_external_power_supply and (con_temp > rapid_temp_shutoff_C):
                    logger.warning(f"Console temperature {con_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    log_line = f"{current_time},SHUTDOWN,Console temperature exceeded rapid shutoff threshold\n"
                    shutdown=True
                elif (tx_temp > rapid_temp_shutoff_C):
                    logger.warning(f"TX device temperature {tx_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded shutoff threshold\n"
                    shutdown=True
                elif (amb_temp > rapid_temp_shutoff_C):
                    logger.warning(f"Ambient temperature {amb_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded shutoff threshold\n"
                    shutdown=True

            log_line = f"{current_time},{frequency_kHz},{duration_msec},{interval_msec},{voltage},{peak_to_peak_voltage},{con_temp},{tx_temp},{amb_temp}\n"
            if not use_external_power_supply:
                logger.info(f"Console Temp: {con_temp}°C TX Temp: {tx_temp}°C Ambient Temp: {amb_temp}°C")
            else:
                logger.info(f"Console Temp: {con_temp} TX Temp: {tx_temp}°C Ambient Temp: {amb_temp}°C")
            logfile.write(log_line)
            logfile.flush()  # Ensure the data is written immediately

            # Check if any temperature exceeds the shutoff threshold
            if not use_external_power_supply and (con_temp > console_shutoff_temp_C):
                logger.warning(f"Console temperature {con_temp}°C exceeds shutoff threshold {console_shutoff_temp_C}°C.")
                log_line = f"{current_time},SHUTDOWN,Console temperature exceeded shutoff threshold\n"
                shutdown=True
            if tx_temp > tx_shutoff_temp_C:
                logger.warning(f"TX device temperature {tx_temp}°C exceeds shutoff threshold {tx_shutoff_temp_C}°C.")
                log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded shutoff threshold\n"
                shutdown=True
            if amb_temp > ambient_shutoff_temp_C:
                logger.warning(f"Ambient temperature {amb_temp}°C exceeds shutoff threshold {ambient_shutoff_temp_C}°C.")
                log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded shutoff threshold\n"
                shutdown=True

            if shutdown:
                interface.txdevice.stop_trigger()
                logfile.write(log_line)
                logfile.flush()  # Ensure the data is written immediately
                break
            time.sleep(log_interval)
    minutes, seconds = divmod(int(time.time() - start), 60)
    logger.info(f"Temperature logging stopped after {minutes}:{seconds:02d}.")
    logger.info(f"Data saved to \"{filename}\".")
    sys.exit(0)

# Verify communication with the devices
if not interface.txdevice.ping():
    logger.error("Failed to ping the transmitter device.")
    sys.exit(1)

if not use_external_power_supply and not interface.hvcontroller.ping():
    logger.error("Failed to ping the console device.")
    sys.exit(1)

if not use_external_power_supply:
    try:
        console_firmware_version = interface.hvcontroller.get_version()
        logger.info(f"Console Firmware Version: {console_firmware_version}")
    except Exception as e:
        logger.error(f"Error querying console firmware version: {e}")

try:
    tx_firmware_version = interface.txdevice.get_version()
    logger.info(f"TX Firmware Version: {tx_firmware_version}")
except Exception as e:
    logger.error(f"Error querying TX firmware version: {e}")

logger.info("Enumerate TX7332 chips")
num_tx_devices = interface.txdevice.enum_tx7332_devices()
if num_tx_devices == 0:
    raise ValueError("No TX7332 devices found.")
elif num_tx_devices == num_modules*2:
    logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
    numelements = 32*num_tx_devices
else:
    raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{num_modules}")

if not use_external_power_supply:
    if interface.hvcontroller.set_voltage(voltage):
        logger.info("High Voltage set successfully.")
    else:
        logger.error("Failed to set High Voltage.")
        sys.exit(1)

pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)

delays = np.zeros(numelements)  # Initialize delays to zero
apodizations = np.ones(numelements)  # Initialize apodizations to ones

test_time_min = 60
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(60/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=test_time_min
)

solution = Solution(
    delays = delays,
    apodizations = apodizations,
    pulse = pulse,
    voltage=voltage,
    sequence = sequence
)

sol_dict = solution.to_dict()
profile_index = 1
profile_increment = True
interface.txdevice.set_solution(
    pulse = sol_dict['pulse'],
    delays = sol_dict['delays'],
    apodizations= sol_dict['apodizations'],
    sequence= sol_dict['sequence'],
    mode="continuous",
    profile_index=profile_index,
    profile_increment=profile_increment
)

duty_cycle = int((duration_msec/interval_msec) * 100)
if duty_cycle > 50:
    logger.warning("❗❗ Duty cycle is above 50% ❗❗")

logger.info(f"User parameters set: \n\
    Frequency: {frequency_kHz}kHz\n\
    Voltage Per Rail: {voltage}V\n\
    Voltage Peak to Peak: {peak_to_peak_voltage}V\n\
    Duration: {duration_msec}ms\n\
    Interval: {interval_msec}ms\n\
    Duty Cycle: {duty_cycle}%\n\
    Use External Power Supply: {use_external_power_supply}\n\
    Initial Temp Safety Shutoff: Increase to {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s of starting.\n\
    General Temp Safety Shutoff: Increase of {rapid_temp_increase_per_second_shutoff_C}°C within {log_interval}s at any point.\n")

logger.info("Press enter to START trigger:")
input()  # Wait for the user to press Enter

if not use_external_power_supply:
    logger.info("Enable  High Voltage")
    if not interface.hvcontroller.turn_hv_on():
        logger.error("Failed to turn on High Voltage.")
        sys.exit(1)
else:
    logger.info("Using external power supply")

def turn_off_console_and_tx():
    if not use_external_power_supply:
        logger.info("Attempting to turn off High Voltage...")
        interface.hvcontroller.turn_hv_off()
        if interface.hvcontroller.get_hv_status():
            logger.error("High Voltage is still on.")
        else:
            logger.info("High Voltage successfully turned off.")

        logger.info("Attempting to turn off 12V...")
        interface.hvcontroller.turn_12v_off()
        if interface.hvcontroller.get_12v_status():
            logger.error("12V is still on.")
        else:
            logger.info("12V successfully turned off.")

shutdown_event = threading.Event()

def input_wrapper():
    if os.name == 'nt':
        while not shutdown_event.is_set():
            if msvcrt.kbhit():  # Check if a key has been pressed
                char = msvcrt.getch()  # Read the key
                if char == b'\r':  # Check if the Enter key was pressed
                    break
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    else:
        while not shutdown_event.is_set():
            # Use select to check if input is available
            ready, _, _ = select.select([], [], [], 0.1)  # Timeout of 0.1 seconds
            if ready:
                sys.stdin.readline() # Read the input
                break

t = threading.Thread(target=log_temperature)
user_input = threading.Thread(target=input_wrapper)

logger.info("Starting Trigger...")
if interface.txdevice.start_trigger():
    logger.info("Trigger Running...")
    logger.info("Press enter to STOP trigger:")

    # Start logging and user input threads
    t.start()
    user_input.start()

    while (user_input.is_alive() and t.is_alive()):
        time.sleep(0.1)

    if (not user_input.is_alive()):
        logger.info("Logging interrupted by user.")
    else:
        logger.info("Logging interrupted by shutdown event.")
        shutdown_event.set()  # Signal the logging thread to stop
    stop_logging = True

    # Wait for threads to finish
    t.join()
    user_input.join()

    time.sleep(0.5)  # Give the logging thread time to finish
    if interface.txdevice.stop_trigger():
        logger.info("Trigger stopped successfully.")
    else:
        logger.error("Failed to stop trigger.")
else:
    logger.error("Failed to get trigger setting.")

turn_off_console_and_tx()
