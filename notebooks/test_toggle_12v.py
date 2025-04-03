from __future__ import annotations

import sys

from openlifu.io.LIFUInterface import LIFUInterface

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

if interface.hvcontroller.is_12v_on:
    interface.hvcontroller.turn_12v_off()
else:
    interface.hvcontroller.turn_12v_on()
