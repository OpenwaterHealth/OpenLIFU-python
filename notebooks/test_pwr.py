from __future__ import annotations

import sys
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_pwr.py

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

print("Ping the device")
interface.hvcontroller.ping()

print("Starting DAC value increments...")
hvp_value = 2400
hrp_value = 2095
hrm_value = 2095
hvm_value = 2400

print(f"Setting DACs: hvp={hvp_value}, hrp={hrp_value}, hvm={hvm_value}, hrm={hrm_value}")
if interface.hvcontroller.set_dacs(hvp=hvp_value, hrp=hrp_value, hvm=hvm_value, hrm=hrm_value):
    print(f"DACs successfully set to hvp={hvp_value}, hrp={hrp_value}, hvm={hvm_value}, hrm={hrm_value}")
else:
    print("Failed to set DACs.")

print("Test HV Supply...")
print("HV OFF. Press enter to TURN ON:")
input()  # Wait for user input
if interface.hvcontroller.turn_hv_on():
    print("HV ON. Press enter to TURN OFF:")
    input()  # Wait for user input
    if interface.hvcontroller.turn_hv_off():
        print("HV OFF.")
    else:
        print("Failed to turn off HV.")
else:
    print("Failed to turn on HV.")
