import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_watertank.py

"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface(test_mode=False)
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

# set focus
xInput = 0
yInput = 0
zInput = 50

frequency = 400e3
voltage = 12.0
duration = 2e-5

pulse = Pulse(frequency=frequency, amplitude=voltage, duration=duration)
pt = Point(position=(xInput,yInput,zInput), units="mm")
sequence = Sequence(
    pulse_interval=0.1,
    pulse_count=10,
    pulse_train_interval=1,
    pulse_train_count=1
)

solution = Solution(
    id="solution",
    name="Solution",
    protocol_id="example_protocol",
    transducer_id="example_transducer",
    delays = np.zeros((1,64)),
    apodizations = np.ones((1,64)),
    pulse = pulse,
    sequence = sequence,
    target=pt,
    foci=[pt],
    approved=True
)

sol_dict = solution.to_dict()
profile_index = 1
profile_increment = True
interface.txdevice.set_solution(
    pulse = sol_dict['pulse'],
    delays = sol_dict['delays'],
    apodizations= sol_dict['apodizations'],
    sequence= sol_dict['sequence'],
    profile_index=profile_index,
    profile_increment=profile_increment
)

print("Get Trigger")
trigger_setting = interface.txdevice.get_trigger_json()
if trigger_setting:
    print(f"Trigger Setting: {trigger_setting}")
else:
    print("Failed to get trigger setting.")

print("Get Temperature")
temperature = interface.txdevice.get_temperature()
print(f"Temperature: {temperature} °C")

print("Get Ambient")
a_temp = interface.txdevice.get_ambient_temperature()
print(f"Ambient Temperature: {a_temp} °C")

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
