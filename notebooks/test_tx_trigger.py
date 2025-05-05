from __future__ import annotations

import sys
import time

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

    while True:
        params = get_user_input()
        if params is None:
            print("Exiting...")
            break

        json_trigger_data = {
            "TriggerFrequencyHz": params["freq"],
            "TriggerPulseCount": 1,
            "TriggerPulseWidthUsec": params["pulse_width"],
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
            continue

        if interface.txdevice.start_trigger():
            print("Trigger Running. Press Enter to STOP:")
            input()  # Wait for the user to press Enter
            if interface.txdevice.stop_trigger():
                print("Trigger stopped successfully.")
            else:
                print("Failed to stop trigger.")

if __name__ == "__main__":
    main()
