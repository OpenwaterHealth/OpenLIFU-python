from __future__ import annotations

import sys
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_async_mode.py

def main():
    print("Starting LIFU Async Test Script...")
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

    curr_mode = interface.txdevice.async_mode()
    print(f"Current Async Mode: {curr_mode}")
    time.sleep(1)
    if curr_mode:
        print("Async mode is already enabled.")
    else:
        print("Enabling Async Mode...")
        interface.txdevice.async_mode(True)
        time.sleep(1)
        curr_mode = interface.txdevice.async_mode()
        print(f"Async Mode Enabled: {curr_mode}")
    time.sleep(1)
    print("Disabling Async Mode...")
    interface.txdevice.async_mode(False)
    time.sleep(1)
    curr_mode = interface.txdevice.async_mode()
    print(f"Async Mode Enabled: {curr_mode}")

if __name__ == "__main__":
    main()
