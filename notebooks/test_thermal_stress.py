from __future__ import annotations

import sys
import threading
import time

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

log_interval = 2  # seconds; you can adjust this variable as needed

frequency_kHz = 400 # Frequency in kHz
voltage = 100.0 # Voltage in Volts
duration_msec = 10 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system

console_shutoff_temp_C = 70.0 # Console shutoff temperature in Celsius
tx_shutoff_temp_C = 70.0 # TX device shutoff temperature in Celsius
ambient_shutowff_temp_C = 70.0 # Ambient shutoff temperature in Celsius

print("Starting LIFU Test Script...")
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()

if not tx_connected:
    print("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()

    # Give time for the TX device to power up and enumerate over USB
    time.sleep(2)

    # Cleanup and recreate interface to reinitialize USB devices
    interface.stop_monitoring()
    del interface
    time.sleep(1)  # Short delay before recreating

    print("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface()

    # Re-check connection
    tx_connected, hv_connected = interface.is_device_connected()

if tx_connected and hv_connected:
    print("✅ LIFU Device fully connected.")
else:
    print("❌ LIFU Device NOT fully connected.")
    print(f"  TX Connected: {tx_connected}")
    print(f"  HV Connected: {hv_connected}")
    sys.exit(1)

# Ask the user if they want to log temperature
log_choice = input("Do you want to log temperature before starting trigger? (y/n): ").strip().lower()
log_temp = (log_choice == "y")
stop_logging = False  # flag to signal the logging thread to stop

def log_temperature():
    # Create a file with the current timestamp in the name
    start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_temp.csv"
    shutdown = False
    with open(filename, "w") as logfile:
        while not (stop_logging or shutdown):
            con_temp = interface.hvcontroller.get_temperature1()
            tx_temp = interface.txdevice.get_temperature()
            amb_temp = interface.txdevice.get_ambient_temperature()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{current_time},{frequency_kHz},{duration_msec},{voltage},{con_temp},{tx_temp},{amb_temp}\n"
            logfile.write(log_line)
            logfile.flush()  # Ensure the data is written immediately
            # Check if any temperature exceeds the shutoff threshold
            if con_temp > console_shutoff_temp_C:
                print(f"Console temperature {con_temp} °C exceeds shutoff threshold {console_shutoff_temp_C} °C.")
                log_line = f"{current_time},SHUTDOWN,Console temperature exceeded shutoff threshold\n"
                shutdown=True
            elif tx_temp > tx_shutoff_temp_C:
                print(f"TX device temperature {tx_temp} °C exceeds shutoff threshold {tx_shutoff_temp_C} °C.")
                log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded shutoff threshold\n"
                shutdown=True
            elif amb_temp > ambient_shutowff_temp_C:
                print(f"Ambient temperature {amb_temp} °C exceeds shutoff threshold {ambient_shutowff_temp_C} °C.")
                log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded shutoff threshold\n"
                shutdown=True
            else:
                shutdown=False
            if shutdown:
                interface.txdevice.stop_trigger()
                logfile.write(log_line)
                logfile.flush()  # Ensure the data is written immediately
                break
            time.sleep(log_interval)
    print(f"Temperature logging stopped after {time.time() - start:.2f} seconds. Data saved to {filename}.")

# Verify communication with the devices
if not interface.txdevice.ping():
    print("Failed to ping the transmitter device.")
    sys.exit(1)

if not interface.hvcontroller.ping():
    print("Failed to ping the console devie.")
    sys.exit(1)

print("Enumerate TX7332 chips")
num_tx_devices = interface.txdevice.enum_tx7332_devices()
if num_tx_devices == 0:
    raise Exception("No TX7332 devices found.")
elif num_tx_devices == num_modules*2:
    print(f"Number of TX7332 devices found: {num_tx_devices}")
    numelements = 32*num_tx_devices
else:
    raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{num_modules}")

print("Set High Voltage")
if interface.hvcontroller.set_voltage(voltage):
    print("High Voltage set successfully.")
else:
    print("Failed to set High Voltage.")
    sys.exit(1)

pulse = Pulse(frequency=frequency_kHz*1e3, amplitude=voltage, duration=duration_msec*1e-3)

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

# If logging is enabled, start the logging thread
if log_temp:
    t = threading.Thread(target=log_temperature)
else:
    print("Get Temperature")
    temperature = interface.txdevice.get_temperature()
    print(f"Temperature: {temperature} °C")

    print("Get Ambient")
    a_temp = interface.txdevice.get_ambient_temperature()
    print(f"Ambient Temperature: {a_temp} °C")

print("Press enter to START trigger:")
input()  # Wait for the user to press Enter

print("Enable  High Voltage")
if not interface.hvcontroller.turn_hv_on():
    print("Failed to turn on High Voltage.")
    sys.exit(1)

print("Starting Trigger...")
if interface.txdevice.start_trigger():
    if log_temp:
        t.start()  # Start the logging thread

    print("Trigger Running Press enter to STOP:")
    input()  # Wait for the user to press Enter
    stop_logging = True
    time.sleep(0.5)  # Give the logging thread time to finish
    if interface.txdevice.stop_trigger():
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")

# Stop the temperature logging before starting the trigger
if log_temp:
    t.join()
