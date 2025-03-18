from __future__ import annotations

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_nucleo.py
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

print("Toggle LED")
interface.txdevice.toggle_led()

print("Get Version")
version = interface.txdevice.get_version()
print(f"Version: {version}")

print("Echo Data")
echo, echo_len = interface.txdevice.echo(echo_data=b'Hello LIFU!')
if echo_len > 0:
    print(f"Echo: {echo.decode('utf-8')}")  # Echo: Hello LIFU!
else:
    print("Echo failed.")

print("Get HW ID")
hw_id = interface.txdevice.get_hardware_id()
print(f"HWID: {hw_id}")

print("Get Temperature")
temperature = interface.txdevice.get_temperature()
print(f"Temperature: {temperature} °C")

print("Get Ambient")
temperature = interface.txdevice.get_ambient_temperature()
print(f"Ambient Temperature: {temperature} °C")

print("Get Trigger")
current_trigger_setting = interface.txdevice.get_trigger_json()
if current_trigger_setting:
    print(f"Current Trigger Setting: {current_trigger_setting}")
else:
    print("Failed to get current trigger setting.")

print("Starting Trigger with current setting...")
if interface.txdevice.start_trigger():
    print("Trigger Running Press enter to STOP:")
    input()  # Wait for the user to press Enter
    if interface.txdevice.stop_trigger():
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")

print("Set Trigger")
json_trigger_data = {
    "TriggerFrequencyHz": 25,
    "TriggerPulseCount": 0,
    "TriggerPulseWidthUsec": 20000,
    "TriggerPulseTrainInterval": 0,
    "TriggerPulseTrainCount": 0,
    "TriggerMode": 1,
    "ProfileIndex": 0,
    "ProfileIncrement": 0
}

trigger_setting = interface.txdevice.set_trigger_json(data=json_trigger_data)
if trigger_setting:
    print(f"Trigger Setting: {trigger_setting}")
else:
    print("Failed to set trigger setting.")

print("Starting Trigger with updated setting...")
if interface.txdevice.start_trigger():
    print("Trigger Running Press enter to STOP:")
    input()  # Wait for the user to press Enter
    if interface.txdevice.stop_trigger():
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")

print("Reset Device:")
# Ask the user for confirmation
user_input = input("Do you want to reset the device? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.txdevice.soft_reset():
        print("Reset Successful.")
elif user_input == 'n':
    print("Reset canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")

print("Update Device:")
# Ask the user for confirmation
user_input = input("Do you want to update the device? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.txdevice.enter_dfu():
        print("Entering DFU Mode.")
elif user_input == 'n':
    print("Update canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")
