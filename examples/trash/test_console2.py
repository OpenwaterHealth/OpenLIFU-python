from __future__ import annotations

import sys

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_console.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface(TX_test_mode=False)
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

if not hv_connected:
       print("HV Controller not connected.")
       sys.exit()

print("Ping the device")
interface.hvcontroller.ping()

# Set High Voltage Level
if not interface.hvcontroller.set_voltage(voltage=30.0):
    print("Failed to set voltage.")

# Get Set High Voltage Setting
print("Get Current HV Voltage")
read_voltage = interface.hvcontroller.get_voltage()
print(f"HV Voltage {read_voltage} V.")


print("Test HV Supply...")
if interface.hvcontroller.turn_hv_on():
# Get Set High Voltage Setting
    read_voltage = interface.hvcontroller.get_voltage()
    print(f"HV Voltage {read_voltage} V.")
    print("HV ON Press enter to TURN OFF:")
    input()  # Wait for the user to press Enter
    if interface.hvcontroller.turn_hv_off():
        print("HV OFF.")
    else:
        print("Failed to turn off HV")
else:
    print("Failed to turn on HV.")

print("Reset DevConsoleice:")
# Ask the user for confirmation
user_input = input("Do you want to reset the Console? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.hvcontroller.soft_reset():
        print("Reset Successful.")
elif user_input == 'n':
    print("Reset canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")
