from __future__ import annotations

import sys
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/demo.py

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

    if hv_connected:
        console_version = interface.hvcontroller.get_version()
        print(f"Version: {console_version}")
        print("HV Controller connected.")
        interface.hvcontroller.set_voltage(24.5)
        interface.hvcontroller.turn_hv_on()
        input("Press Enter to continue...")

        interface.hvcontroller.turn_hv_off()



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

        print("HV Controller connected.")
        interface.hvcontroller.set_voltage(24.5)
        interface.hvcontroller.turn_hv_on()

    device_count = interface.txdevice.get_tx_module_count()
    print(f"Device Count: {device_count}")

    for i in range(1, device_count + 1):

        print("Ping the device")
        if not interface.txdevice.ping(module=i):
            print("❌ Failed comms with txdevice.")
            sys.exit(1)

        version = interface.txdevice.get_version(module=i)
        print(f"Version: {version}")


    if hv_connected:
        time.sleep(5)
        print("disabling HV for safety...")
        interface.hvcontroller.turn_hv_off()

if __name__ == "__main__":
    main()
