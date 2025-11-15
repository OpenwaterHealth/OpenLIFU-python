from __future__ import annotations

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_updated_pwr.py
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
interface.hvcontroller.ping()

print("Toggle LED")
interface.hvcontroller.toggle_led()

print("Get Version")
version = interface.hvcontroller.get_version()
print(f"Version: {version}")

print("Echo Data")
echo, echo_len = interface.hvcontroller.echo(echo_data=b'Hello LIFU!')
if echo_len > 0:
    print(f"Echo: {echo.decode('utf-8')}")  # Echo: Hello LIFU!
else:
    print("Echo failed.")

print("Get HW ID")
hw_id = interface.hvcontroller.get_hardware_id()
print(f"HWID: {hw_id}")

print("Test 12V...")
if interface.hvcontroller.turn_12v_on():
    print("12V ON Press enter to TURN OFF:")
    input()  # Wait for the user to press Enter
    if interface.hvcontroller.turn_12v_off():
        print("12V OFF.")
    else:
        print("Failed to turn off 12V")
else:
    print("Failed to turn on 12V.")

# Set High Voltage Level
print("Set HV Power to +/- 24V")
if interface.hvcontroller.set_voltage(voltage=24.0):
    print("Voltage set to 24.0 V.")
else:
    print("Failed to set voltage.")

# Get Set High Voltage Setting
print("Get HV Setting")
read_set_voltage = interface.hvcontroller.get_voltage()
print(f"Voltage set to {read_set_voltage} V.")


print("Test HV Supply...")
if interface.hvcontroller.turn_hv_on():
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
