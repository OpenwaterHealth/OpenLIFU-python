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

# Increment DAC values from 0 to 3000 by 100, with reg +400 capped at 4095
for dac_value in range(0, 3001, 100):
    reg_value = min(dac_value + 400, 4095)
    print(f"Setting DACs: hvp={dac_value}, hrp={reg_value}, hvm={dac_value}, hrm={reg_value}")
    if interface.hvcontroller.set_dacs(hvp=dac_value, hrp=reg_value, hvm=dac_value, hrm=reg_value):
        print(f"DACs successfully set to hvp={dac_value}, hrp={reg_value}, hvm={dac_value}, hrm={reg_value}")
    else:
        print("Failed to set DACs.")

    print("Test HV Supply...")
    if interface.hvcontroller.turn_hv_on():
        print("HV ON. Press enter to TURN OFF:")
        input()  # Wait for user input
        if interface.hvcontroller.turn_hv_off():
            print("HV OFF.")
        else:
            print("Failed to turn off HV.")
    else:
        print("Failed to turn on HV.")
