# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# # 04: Connecting to OpenLIFU Hardware
#
# This notebook explains how to establish a connection with OpenLIFU hardware using the `LIFUInterface` class. It covers checking device status, pinging the devices, and retrieving basic information like firmware versions and hardware IDs.
#
# **Note:** To run this notebook effectively, you need OpenLIFU hardware connected to your computer.

# ## 1. Imports and `LIFUInterface`
#
# The primary class for interacting with the hardware is `LIFUInterface` from `openlifu.io`.
from __future__ import annotations

import time

from openlifu.io.LIFUInterface import LIFUInterface

# ### Instantiating `LIFUInterface`
#
# When you create an instance of `LIFUInterface`, it attempts to automatically detect and connect to the OpenLIFU hardware components:
# *   **TX Device (Transmitter Controller):** Manages the ultrasound transmission, including waveform generation, delays, and triggering. Accessed via `interface.txdevice`.
# *   **HV Controller (Console):** Manages power supplies (12V, High Voltage), temperature monitoring, fans, etc. Accessed via `interface.hvcontroller`.
#
# By default, the interface runs in a synchronous mode. Asynchronous operation for background monitoring and events will be covered in a later notebook.

try:
    print("Initializing LIFUInterface...")
    # The interface will try to connect to available serial ports matching OpenLIFU devices.
    # If devices are not found immediately, it might take a moment or require them to be powered on.
    interface = LIFUInterface()
    print("LIFUInterface initialized.")
except Exception as e:
    print(f"Error initializing LIFUInterface: {e}")
    print("Please ensure OpenLIFU hardware is connected and powered on.")
    interface = None # Set to None if initialization fails

# ## 2. Checking Device Connection Status
#
# The `is_device_connected()` method tells you if the TX device and HV controller are currently connected.

if interface:
    # It can take a moment for devices to be fully enumerated and connected,
    # especially if they were just powered on or if 12V needs to be enabled for the TX.
    print("\nChecking initial device connection status...")
    time.sleep(1) # Give a brief moment for connections to establish
    tx_connected, hv_connected = interface.is_device_connected()

    print(f"  Transmitter (TX) Connected: {tx_connected}")
    print(f"  HV Controller (HV) Connected: {hv_connected}")

    if tx_connected and hv_connected:
        print("✅ Both TX and HV Controller are connected.")
    elif tx_connected:
        print("🟡 TX connected, but HV Controller is NOT connected.")
        print("   Some operations, especially power control, will not be available.")
    elif hv_connected:
        print("🟡 HV Controller connected, but TX is NOT connected.")
        print("   Ensure 12V power is enabled via the HV Controller if it controls TX power.")
        print("   (See Notebook 06 for controlling 12V power).")
    else:
        print("❌ Neither TX nor HV Controller are connected.")
        print("   Please check hardware connections and power.")
else:
    print("LIFUInterface not initialized. Cannot check connections.")

# ## 3. Pinging Devices
#
# Pinging is a basic way to check if the connected devices are responsive.

if interface:
    if tx_connected:
        print("\nPinging TX Device...")
        try:
            if interface.txdevice.ping():
                print("  ✅ TX Device responded to ping.")
            else:
                print("  ❌ TX Device did NOT respond to ping or an error occurred.")
        except Exception as e:
            print(f"  ⚠️ Error pinging TX Device: {e}")
    else:
        print("\nTX Device not connected, skipping ping.")

    if hv_connected:
        print("\nPinging HV Controller...")
        try:
            if interface.hvcontroller.ping():
                print("  ✅ HV Controller responded to ping.")
            else:
                print("  ❌ HV Controller did NOT respond to ping or an error occurred.")
        except Exception as e:
            print(f"  ⚠️ Error pinging HV Controller: {e}")
    else:
        print("\nHV Controller not connected, skipping ping.")
else:
    print("LIFUInterface not initialized. Cannot ping devices.")

# ## 4. Retrieving Firmware Versions
#
# You can get the firmware versions of the connected devices.

if interface:
    if tx_connected:
        print("\nGetting TX Device Firmware Version...")
        try:
            version_tx = interface.txdevice.get_version()
            print(f"  TX Device Firmware Version: {version_tx}")
        except Exception as e:
            print(f"  ⚠️ Error getting TX Device firmware version: {e}")
    else:
        print("\nTX Device not connected, cannot get firmware version.")

    if hv_connected:
        print("\nGetting HV Controller Firmware Version...")
        try:
            version_hv = interface.hvcontroller.get_version()
            print(f"  HV Controller Firmware Version: {version_hv}")
        except Exception as e:
            print(f"  ⚠️ Error getting HV Controller firmware version: {e}")
    else:
        print("\nHV Controller not connected, cannot get firmware version.")
else:
    print("LIFUInterface not initialized. Cannot get firmware versions.")

# ## 5. Retrieving Hardware IDs
#
# Devices often have unique hardware identifiers.

if interface:
    if tx_connected:
        print("\nGetting TX Device Hardware ID...")
        try:
            # Note: The method to get hardware ID might vary or might not be implemented for all devices.
            # Assuming a hypothetical get_hardware_id() method for txdevice.
            # Check the specific documentation for interface.txdevice if this fails.
            if hasattr(interface.txdevice, 'get_hardware_id'):
                hw_id_tx = interface.txdevice.get_hardware_id()
                print(f"  TX Device Hardware ID: {hw_id_tx}")
            elif hasattr(interface.txdevice, 'get_uid_ Polize'): # common alternative name
                hw_id_tx = interface.txdevice.get_uid_str()
                print(f"  TX Device UID: {hw_id_tx}")
            else:
                print("  TX Device does not have a standard 'get_hardware_id()' or 'get_uid_str()' method in this example.")
        except Exception as e:
            print(f"  ⚠️ Error getting TX Device hardware ID: {e}")
    else:
        print("\nTX Device not connected, cannot get hardware ID.")

    if hv_connected:
        print("\nGetting HV Controller Hardware ID...")
        try:
            hw_id_hv = interface.hvcontroller.get_hardware_id() # This method exists in test_console.py
            print(f"  HV Controller Hardware ID: {hw_id_hv}")
            # The ID is often presented in base58 for console
            import base58
            if hw_id_hv and isinstance(hw_id_hv, str) and len(hw_id_hv) == 16 : # Expected format from console
                 encoded_id = base58.b58encode(bytes.fromhex(hw_id_hv)).decode()
                 print(f"  HV Controller Full ID (example format): OW-LIFU-CON-{encoded_id}")

        except Exception as e:
            print(f"  ⚠️ Error getting HV Controller hardware ID: {e}")
    else:
        print("\nHV Controller not connected, cannot get hardware ID.")
else:
    print("LIFUInterface not initialized. Cannot get hardware IDs.")

# ## 6. Important Note on TX Device Power
#
# For some OpenLIFU setups, the TX device receives its main 12V power through the HV Controller.
# If the TX device is not connecting, you might first need to connect to the HV Controller and explicitly turn on the 12V supply. This will be covered in Notebook 06, which focuses on HV Controller functionalities.
#
# If 12V is turned on, you might need to re-initialize `LIFUInterface` or wait for it to detect the newly powered TX device.
# ```python
# # Example:
# # if interface and interface.hvcontroller.is_connected():
# #     print("Turning on 12V power via HV Controller...")
# #     interface.hvcontroller.turn_12v_on()
# #     time.sleep(2) # Give time for TX to power up
# #     # Re-check connections or even re-initialize interface if txdevice doesn't appear
# #     del interface
# #     interface = LIFUInterface()
# #     tx_connected, _ = interface.is_device_connected()
# #     print(f"TX connected after 12V power on: {tx_connected}")
# ```

# ## 7. Cleaning Up
#
# When you're done interacting with the hardware, it's good practice to clean up the interface, especially if it's running in asynchronous mode (though for synchronous mode, Python's garbage collection usually handles it).
# For synchronous mode, explicitly deleting the object or letting it go out of scope is often sufficient.
# If background threads were started (not the default for LIFUInterface unless `run_async=True`), they should be stopped.

if interface:
    print("\nFinished with hardware interaction for this notebook.")
    # For synchronous interface, explicit cleanup might not be strictly necessary
    # but if there were any persistent connections or threads (not typical for default init):
    # interface.stop_monitoring() # Important if async monitoring was started
    del interface
    print("LIFUInterface instance deleted.")

# ## Next Steps
#
# Now that you know how to connect to the hardware and verify basic communication, you're ready to send a calculated `Solution` to the device.
#
# *   **Notebook 05 (`05_Solution_to_Hardware_Basic.py`):** Will demonstrate how to take a `Solution` (like one calculated in Notebook 03) and send it to the TX device to be programmed for sonication.
# *   **Notebook 06 (`06_Hardware_Console_Controls.py`):** Will dive deeper into controlling the HV Controller, including power management (12V, HV), fans, and temperature readings.

# End of Notebook 04
