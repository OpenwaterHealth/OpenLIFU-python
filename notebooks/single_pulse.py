from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

if os.name == 'nt':
    pass
else:
    pass

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_watertank.py

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
    logger.propagate = False

log_interval = 1  # seconds; you can adjust this variable as needed

# set focus
xInput = 0
yInput = 0
zInput = 50

frequency_kHz = 400 # Frequency in kHz
voltage = 20.0 # Voltage in Volts
duration_msec = 5/400 # Pulse Duration in milliseconds
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

peak_to_peak_voltage = voltage * 2 # Peak to peak voltage for the pulse

here = Path(__file__).parent.resolve()
db_path = here / ".." / "db_dvc"
db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1_005")
arr.sort_by_pin()

target = Point(position=(xInput,yInput,zInput), units="mm")
focus = target.get_position(units="mm")
distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1)).reshape(1,-1)
tof = distances*1e-3 / 1500
delays = tof.max() - tof
print(f"TOF Max = {tof.max()*1e6} us")

apodizations = np.ones((1, arr.numelements()))
#active_element = 25
#active_element = np.arange(65,128)
#apodizations = np.zeros((1, arr.numelements()))
#apodizations[:, active_element-1] = 1

logger.info("Starting LIFU Test Script...")
interface = LIFUInterface(ext_power_supply=use_external_power_supply)
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
    interface =  LIFUInterface(ext_power_supply=use_external_power_supply)

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

logger.info(f'Apodizations: {apodizations}')
logger.info(f'Delays: {delays}')

pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)

sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(60/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=1
)

pin_order = np.argsort([el.pin for el in arr.elements])
solution = Solution(
    delays = delays[:, pin_order],
    apodizations = apodizations[:, pin_order],
    transducer=arr,
    pulse = pulse,
    voltage=voltage,
    sequence = sequence
)
profile_index = 1
profile_increment = True
trigger_mode = "single"

if use_external_power_supply:
    logger.info(f"Using external power supply. Ensure HV is turned on and set to {voltage}V before starting the trigger.")

interface.set_solution(
    solution=solution,
    profile_index=profile_index,
    profile_increment=profile_increment,
    trigger_mode=trigger_mode)

logger.info("Get Trigger")
trigger_setting = interface.txdevice.get_trigger_json()
if trigger_setting:
    logger.info(f"Trigger Setting: {trigger_setting}")
else:
    logger.error("Failed to get trigger setting.")
    sys.exit(1)

duty_cycle = int((duration_msec/interval_msec) * 100)
if duty_cycle > 50:
    logger.warning("❗❗ Duty cycle is above 50% ❗❗")

logger.info(f"User parameters set: \n\
    Module Invert: {arr.module_invert}\n\
    Frequency: {frequency_kHz}kHz\n\
    Voltage Per Rail: {voltage}V\n\
    Voltage Peak to Peak: {peak_to_peak_voltage}V\n\
    Duration: {duration_msec}ms\n\
    Interval: {interval_msec}ms\n\
    Duty Cycle: {duty_cycle}%\n\
    Use External Power Supply: {use_external_power_supply}\n\
    Initial Temp Safety Shutoff: Increase to {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s of starting.\n\
    General Temp Safety Shutoff: Increase of {rapid_temp_increase_per_second_shutoff_C}°C within {log_interval}s at any point.\n")

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

logger.info("turning HV ON")
interface.hvcontroller.turn_hv_on()
s = input("Press any key to start")
logger.info("Sending Single Trigger...")
interface.txdevice.start_trigger()
time.sleep(0.1)
interface.txdevice.stop_trigger()
interface.hvcontroller.turn_hv_off()
turn_off_console_and_tx()
logger.info("Finished")
