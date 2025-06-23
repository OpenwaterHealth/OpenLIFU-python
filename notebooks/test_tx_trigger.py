from __future__ import annotations

import sys
import time

from openlifu.io.LIFUConfig import (
    TRIGGER_MODE_SEQUENCE,
    TRIGGER_MODE_SINGLE,
)
from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_tx_trigger.py

def get_user_input():
    while True:
        print("\nEnter parameters (or 'x' to exit):")
        try:
            freq = input("Trigger Frequency (Hz): ")
            if freq.lower() == 'x':
                return None

            pulse_width = input("Pulse Width (μs): ")
            if pulse_width.lower() == 'x':
                return None

            return {
                "freq": float(freq),
                "pulse_width": float(pulse_width)
            }
        except ValueError:
            print("Invalid input. Please enter numbers or 'x' to exit.")

def main():
    print("Starting LIFU Test Script...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()

    if not tx_connected and not hv_connected:
        print("✅ LIFU Console not connected.")
        sys.exit(1)

    if not tx_connected:
        print("TX device not connected. Attempting to turn on 12V...")
        interface.hvcontroller.turn_12v_on()
        time.sleep(2)

        interface.stop_monitoring()
        del interface
        time.sleep(3)

        print("Reinitializing LIFU interface after powering 12V...")
        interface = LIFUInterface()
        tx_connected, hv_connected = interface.is_device_connected()

        if tx_connected and hv_connected:
            print("✅ LIFU Device fully connected.")
        else:
            print("❌ LIFU Device NOT fully connected.")
            print(f"  TX Connected: {tx_connected}")
            print(f"  HV Connected: {hv_connected}")
            sys.exit(1)

    print("Ping the device")
    if not interface.txdevice.ping():
        print("❌ Failed comms with txdevice.")
        sys.exit(1)

    version = interface.txdevice.get_version()
    print(f"Version: {version}")

    while True:
        params = get_user_input()
        if params is None:
            print("Exiting...")
            if interface.txdevice.is_connected:
                print("Disconnecting TX device...")
                interface.txdevice.close()

            break

        json_trigger_data = {
            "TriggerFrequencyHz": params["freq"],
            "TriggerPulseCount": 10,
            "TriggerPulseWidthUsec": params["pulse_width"],
            "TriggerPulseTrainInterval": 300000,
            "TriggerPulseTrainCount": 10,
            "TriggerMode": TRIGGER_MODE_SEQUENCE, # Change to TRIGGER_MODE_CONTINUOUS or TRIGGER_MODE_SEQUENCE or TRIGGER_MODE_SINGLE as needed
            "ProfileIndex": 0,
            "ProfileIncrement": 0
        }

        trigger_setting = interface.txdevice.set_trigger_json(data=json_trigger_data)

        if trigger_setting:
            print(f"Trigger Setting: {trigger_setting}")
        else:
            print("Failed to set trigger setting.")
            continue

        if trigger_setting["TriggerMode"]  == TRIGGER_MODE_SINGLE:
            print("Trigger Mode set to SINGLE. Press Enter to START:")
            if interface.txdevice.start_trigger():
                print("Trigger started successfully.")
            else:
                print("Failed to start trigger.")

        elif trigger_setting["TriggerMode"]  == TRIGGER_MODE_SEQUENCE:
            print("Trigger Mode set to SEQUENCE")
            if interface.txdevice.start_trigger():
                print("Trigger started successfully.")
            else:
                print("Failed to start trigger.")
            while True:
                trigger_status = interface.txdevice.get_trigger_json()
                if trigger_status is None:
                    print("Failed to get trigger status! Fatal Error.")
                    break

                if trigger_status["TriggerStatus"] == "STOPPED":
                    print("Run Complete.")
                    break

                time.sleep(.5)
        else:
            print("Trigger Running Continuous Mode. Press Enter to STOP:")

            if interface.txdevice.start_trigger():
                print("Trigger started successfully.")
            else:
                print("Failed to start trigger.")

            input()  # Wait for the user to press Enter
            if interface.txdevice.stop_trigger():
                print("Trigger stopped successfully.")
            else:
                print("Failed to stop trigger.")

if __name__ == "__main__":
    main()
