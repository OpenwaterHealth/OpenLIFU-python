from __future__ import annotations

import threading
import time

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution
from openlifu.xdc import Transducer

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_watertank.py

"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""

log_interval = 1  # seconds; you can adjust this variable as needed
stop_logging = False  # flag to signal the logging thread to stop

def log_temperature():
    # Create a file with the current timestamp in the name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_temp.log"
    with open(filename, "w") as logfile:
        while not stop_logging:
            temperature = interface.txdevice.get_temperature()
            a_temp = interface.txdevice.get_ambient_temperature()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{current_time}: Temperature: {temperature}, Ambient Temperature: {a_temp}\n"
            logfile.write(log_line)
            logfile.flush()  # Ensure the data is written immediately
            time.sleep(log_interval)


print("Starting LIFU Test Script...")
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

# Ask the user if they want to log temperature
log_choice = input("Do you want to log temperature before starting trigger? (y/n): ").strip().lower()
log_temp = (log_choice == "y")

print("Ping the device")
interface.txdevice.ping()

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

frequency = 405e3
voltage = 12.0
duration = 2e-5

pulse = Pulse(frequency=frequency, amplitude=voltage, duration=duration)
pt = Point(position=(xInput,yInput,zInput), units="mm")

#arr = Transducer.from_file(r"C:\Users\Neuromod2\Documents\OpenLIFU-python\OpenLIFU_2x.json")
# arr = Transducer.from_file(R"..\M4_flex.json")
arr = Transducer.from_file(R"E:\CURRENT-WORK\openwater\OpenLIFU-python\notebooks\pinmap.json")

focus = pt.get_position(units="mm")
#arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1))
tof = distances*1e-3 / 1500
delays = tof.max() - tof
apodizations = np.ones(arr.numelements())



    # tURN only single element ON
#active_element = 25

#delays = delays*0.0
#apodizations = np.zeros(arr.numelements())
#apodizations[active_element-1] = 1
print('apodizations', apodizations)
print('Delays', delays)

sequence = Sequence(
    pulse_interval=0.1,
    pulse_count=10,
    pulse_train_interval=1,
    pulse_train_count=1
)

solution = Solution(
    delays = delays,
    apodizations = apodizations,
    pulse = pulse,
    sequence = sequence
)

sol_dict = solution.to_dict()
profile_index = 1
profile_increment = True
interface.txdevice.set_solution(
    pulse = sol_dict['pulse'],
    delays = sol_dict['delays'],
    apodizations= sol_dict['apodizations'],
    sequence= sol_dict['sequence'],
    mode="continuous",
    profile_index=profile_index,
    profile_increment=profile_increment
)

print("Get Trigger")
trigger_setting = interface.txdevice.get_trigger_json()
if trigger_setting:
    print(f"Trigger Setting: {trigger_setting}")
else:
    print("Failed to get trigger setting.")

# If logging is enabled, start the logging thread
if log_temp:
    t = threading.Thread(target=log_temperature)
    t.start()
else:
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
        stop_logging = True
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")

# Stop the temperature logging before starting the trigger
if log_temp:
    t.join()
