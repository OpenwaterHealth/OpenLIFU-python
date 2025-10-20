from __future__ import annotations

import sys
from time import sleep

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_console_dfu.py
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
       sys.exit(1)

print("Ping the device")
interface.hvcontroller.ping()

# Ask the user for confirmation
user_input = input("Do you want to Enter DFU Mode? (y/n): ").strip().lower()

if user_input == 'y':
    print("Enter DFU mode")
    if interface.hvcontroller.enter_dfu():
        print("Successful.")

        print("Use stm32 cube programmer to update firmware, power cycle will put the console back into an operating state")
        sys.exit(0)

elif user_input == 'n':
    print("Reset device")
    if interface.hvcontroller.soft_reset():
        print("Successful.")

sleep(6)
interface.hvcontroller.uart.reopen_after_reset()
print("Ping the device again")
if  interface.hvcontroller.ping():
    print("Test script complete.")
else:
    print("Device did not respond after reset.")
