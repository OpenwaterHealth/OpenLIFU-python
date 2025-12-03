from __future__ import annotations

import sys
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_tx_dfu.py

print("Starting LIFU Test Script...")
interface = LIFUInterface()

tx_connected, hv_connected = interface.is_device_connected()

if not tx_connected and not hv_connected:
    print("✅ LIFU Console not connected.")
    sys.exit(1)

if not tx_connected:
    print("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()

    # Give time for the TX device to power up and enumerate over USB
    time.sleep(2)

    # Cleanup and recreate interface to reinitialize USB devices
    interface.stop_monitoring()
    del interface
    time.sleep(5)  # Short delay before recreating

    print("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface()

    # Re-check connection
    tx_connected, hv_connected = interface.is_device_connected()

if tx_connected:
    print("✅ LIFU Device TX connected.")
else:
    print("❌ LIFU Device NOT fully connected.")
    print(f"  TX Connected: {tx_connected}")
    print(f"  HV Connected: {hv_connected}")
    sys.exit(1)


print("Ping the device")
if not interface.txdevice.ping():
    print("❌ failed to communicate with transmit module")
    sys.exit(1)

print("Get Version")
version = interface.txdevice.get_version()
print(f"Version: {version}")


# Ask the user for confirmation
user_input = input("Do you want to Enter DFU Mode? (y/n): ").strip().lower()

if user_input == 'y':
    print("Enter DFU mode")
    if interface.txdevice.enter_dfu():
        print("Successful.")

        print("Use stm32 cube programmer to update firmware, power cycle will put the console back into an operating state")
        sys.exit(0)

elif user_input == 'n':
    print("Reset device")
    if interface.txdevice.soft_reset():
        print("Successful.")

time.sleep(6)
interface.txdevice.uart.reopen_after_reset()
print("Ping the device again")
if  interface.txdevice.ping():
    print("Test script complete.")
else:
    print("Device did not respond after reset.")
