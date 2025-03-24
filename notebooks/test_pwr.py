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

print("Starting DAC value increments...")
dac_value = 2800
reg_value = 3800
print(f"Setting DACs: hvp={dac_value}, hrp={reg_value}, hvm={dac_value}, hrm={reg_value}")
if interface.hvcontroller.set_dacs(hvp=dac_value, hrp=reg_value, hvm=dac_value, hrm=reg_value):
    print(f"DACs successfully set to hvp={dac_value}, hrp={reg_value}, hvm={dac_value}, hrm={reg_value}")
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
