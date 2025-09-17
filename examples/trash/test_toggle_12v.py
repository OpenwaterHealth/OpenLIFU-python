from __future__ import annotations

import sys

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_toggle_12v.py

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

interface.hvcontroller.turn_12v_on()

print("12v ON. Press enter to TURN OFF:")
input()  # Wait for user input

interface.hvcontroller.turn_12v_off()
