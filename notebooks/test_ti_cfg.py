from __future__ import annotations

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_ti_cfg.py
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

print("Get Temperature")
temperature = interface.txdevice.get_temperature()
print(f"Temperature: {temperature} °C")

print("Enumerate TX7332 chips")
num_tx_devices = interface.txdevice.enum_tx7332_devices()
if num_tx_devices > 0:
    print(f"Number of TX7332 devices found: {num_tx_devices}")
else:
    raise Exception("No TX7332 devices found.")

print("Set TX7332 TI Config Waveform")
for idx in range(num_tx_devices):
    interface.txdevice.apply_ti_config_file(txchip_id=idx, file_path="notebooks/ti_example.cfg")

print("Get Trigger")
trigger_setting = interface.txdevice.get_trigger_json()
if trigger_setting:
    print(f"Trigger Setting: {trigger_setting}")
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

print("Press enter to START trigger:")
input()  # Wait for the user to press Enter
print("Starting Trigger...")
if interface.txdevice.start_trigger():
    print("Trigger Running Press enter to STOP:")
    input()  # Wait for the user to press Enter
    if interface.txdevice.stop_trigger():
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")
