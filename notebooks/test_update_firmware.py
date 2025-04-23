from __future__ import annotations

import sys

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_update_firmware.py
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

print("Enter DFU mode")
interface.hvcontroller.enter_dfu()

print("Use stm32 cube programmer to update firmware, power cycle will put the console back into an operating state")
sys.exit(0)
