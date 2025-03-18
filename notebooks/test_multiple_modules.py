from __future__ import annotations

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_multiple_modules.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

print("Ping the device")
interface.txdevice.ping()

print("Enumerate TX7332 chips")
num_tx_devices = interface.txdevice.enum_tx7332_devices()
if num_tx_devices > 0:
    print(f"Number of TX7332 devices found: {num_tx_devices}")

    print("Write Demo Registers to TX7332 chips")
    for device_index in range(num_tx_devices):
        interface.txdevice.demo_tx7332(device_index)

    print("Starting Trigger...")
    if interface.txdevice.start_trigger():
        print("Trigger Running Press enter to STOP:")
        input()  # Wait for the user to press Enter
        if interface.txdevice.stop_trigger():
            print("Trigger stopped successfully.")
        else:
            print("Failed to stop trigger.")
    else:
        print("Failed to start trigger.")

else:
    raise Exception("No TX7332 devices found.")
