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

# # 07: Hardware Transmitter (TX) Device Controls
#
# This notebook focuses on direct interactions with the OpenLIFU Transmitter (TX) device. The TX device is responsible for generating the precise electronic signals that drive the ultrasound transducer elements, based on the programmed solution.
#
# We'll cover:
# *   Basic communication checks (ping, version).
# *   Running on-board self-tests.
# *   Reading temperatures from TX device sensors.
# *   Enumerating TX7332 driver chips (important for multi-module systems).

# ## 1. Imports
from __future__ import annotations

from openlifu.io.LIFUInterface import LIFUInterface

# ## 2. Connect to Hardware and Verify TX Device
#
# We initialize `LIFUInterface` and ensure the TX Device is connected.
# Remember from Notebook 06 that the TX device might require 12V power to be enabled via the HV Controller.

# +
interface = None
tx_device = None # Will assign interface.txdevice here

try:
    print("Initializing LIFUInterface...")
    interface = LIFUInterface(HV_test_mode=True) # HV_test_mode=True if only focusing on TX & 12V is already on
    tx_connected, hv_connected = interface.is_device_connected()
    print(f"  TX Connected: {tx_connected}, HV Connected: {hv_connected}")

    if not tx_connected:
        print("⚠️ TX Device is not connected.")
        if hv_connected:
            print("   Ensure 12V power is enabled via the HV Controller if it controls TX power.")
            print("   You may need to run relevant parts of Notebook 06 first, then re-run this cell,")
            print("   or even re-initialize the interface if TX appears after 12V power up.")
            # Example:
            # if interface.hvcontroller.turn_12v_on():
            #    print("   12V turned ON. Waiting for TX to enumerate...")
            #    time.sleep(3)
            #    del interface # Force re-detection
            #    interface = LIFUInterface()
            #    tx_connected, _ = interface.is_device_connected()
            #    if tx_connected: print("   TX device connected after 12V power cycle.")
            #    else: raise ConnectionError("TX Device still not connected after 12V power cycle.")
            # else: raise ConnectionError("Failed to turn on 12V for TX device.")
        else:
            raise ConnectionError("TX Device and HV Controller are not connected.")

    tx_device = interface.txdevice
    print("✅ TX Device is connected and accessible via `tx_device`.")

except Exception as e:
    print(f"Error initializing LIFUInterface or TX Device not connected: {e}")
# -

# ## 3. Basic Information

if tx_device:
    # Ping
    print("\nPinging TX Device...")
    try:
        if tx_device.ping():
            print("  ✅ TX Device responded to ping.")
        else:
            print("  ❌ TX Device did NOT respond or an error occurred.")
    except Exception as e:
        print(f"  ⚠️ Error pinging TX Device: {e}")

    # Get Version
    print("\nGetting TX Device Firmware Version...")
    try:
        version_tx = tx_device.get_version()
        print(f"  TX Device Firmware Version: {version_tx}")
    except Exception as e:
        print(f"  ⚠️ Error getting firmware version: {e}")

    # Get Hardware ID / UID (if available)
    print("\nGetting TX Device Hardware ID/UID (if available)...")
    try:
        # Method names for hardware ID can vary. Common ones are get_hardware_id() or get_uid_str().
        if hasattr(tx_device, 'get_hardware_id'):
            hw_id_tx = tx_device.get_hardware_id()
            print(f"  TX Device Hardware ID: {hw_id_tx}")
        elif hasattr(tx_device, 'get_uid_str'): # A common alternative
            uid_tx = tx_device.get_uid_str()
            print(f"  TX Device UID: {uid_tx}")
        else:
            print("  Standard method for Hardware ID/UID not found on this TX device object.")
    except Exception as e:
        print(f"  ⚠️ Error getting hardware ID/UID: {e}")

# ## 4. Running Self-Tests
# The TX device often has built-in self-tests to check its own circuitry.

if tx_device:
    print("\nRunning TX Device Self-Test...")
    try:
        # The run_test() method's return value and output can vary.
        # Some might return True/False, others might print results or raise errors.
        test_result = tx_device.run_test() # From run_self_test.py, it doesn't seem to return a value directly.
                                          # It might print to console or log.
        print(f"  Self-test command sent. Result (if any): {test_result}")
        print("  Check device logs or console output for detailed self-test results if not directly returned.")
        # If the test is known to take time, add a small delay
        # time.sleep(1)
    except Exception as e:
        print(f"  ⚠️ Error running self-test: {e}")

# ## 5. Temperature Monitoring (TX Device Sensors)
# The TX device usually has its own temperature sensors.

if tx_device:
    print("\nReading Temperatures from TX Device Sensors...")
    # Main TX Board Temperature
    try:
        tx_temp = tx_device.get_temperature()
        print(f"  TX Device Main Temperature: {tx_temp}°C")
    except Exception as e:
        print(f"  ⚠️ Error reading TX Device main temperature: {e}")

    # Ambient Temperature (often on the TX board or nearby)
    try:
        ambient_temp = tx_device.get_ambient_temperature()
        print(f"  TX Device Ambient Temperature: {ambient_temp}°C")
    except Exception as e:
        print(f"  ⚠️ Error reading TX Device ambient temperature: {e} (Sensor might not be available).")

# ## 6. Enumerating TX7332 Chips
#
# Modern OpenLIFU systems might use one or more TI TX7332 chips (or similar) to drive transducer elements.
# Enumerating these chips helps understand the hardware configuration, especially for multi-module systems.
# Each TX7332 typically drives a certain number of channels (e.g., 32 channels).

if tx_device:
    print("\nEnumerating TX7332 Driver Chips...")
    try:
        num_tx_chips = tx_device.enum_tx7332_devices()
        if num_tx_chips > 0:
            print(f"  ✅ Found {num_tx_chips} TX7332 driver chip(s) on the TX device.")
            # For a system with e.g. 64 elements driven by two 32-channel chips, num_tx_chips would be 2.
            # For a system with e.g. 128 elements driven by four 32-channel chips, num_tx_chips would be 4.
            # This implies the system can control num_tx_chips * channels_per_chip elements.
        elif num_tx_chips == 0:
            print("  No TX7332 driver chips reported. This might be normal for some TX board types,")
            print("  or could indicate an issue if they are expected.")
        else: # Should not happen, usually 0 or positive
            print(f"  Unexpected result from enum_tx7332_devices: {num_tx_chips}")

    except Exception as e:
        print(f"  ⚠️ Error enumerating TX7332 chips: {e}")
        print("     (This TX device might not use TX7332 chips or the command failed).")

# ## 7. Note on Register Access
#
# While the TX device's operation is controlled by various internal registers, direct register access is typically an advanced operation used for debugging or very specific low-level configurations.
#
# Most standard operations (like applying a solution, triggering, self-tests) are handled by higher-level methods of the `tx_device` object.
#
# If you need to inspect or modify registers, you would use methods like:
# `tx_device.tx_registers.read_register(chip_index, address)`
# `tx_device.tx_registers.write_register(chip_index, address, value)`
# (The exact methods and `tx_registers` object structure might vary based on the specific TX board driver in OpenLIFU.)
#
# Detailed register maps are usually found in the TX7332 datasheet or the TX board's hardware documentation.
# Notebook `test_solution.py` (in its original form) had an example of reading many registers, which could be adapted for advanced use.

# ## 8. Cleanup

if interface:
    print("\nFinished with TX Device interactions for this notebook.")
    del interface
    interface = None
    tx_device = None
    print("LIFUInterface instance deleted.")

# ## Next Steps
#
# This notebook covered direct interactions with the TX (Transmitter) device.
#
# *   **Recap:** We've now seen how to generate solutions (NB 03), connect to hardware (NB 04), send solutions and trigger (NB 05), control the console/HV power (NB 06), and interact with the TX board (this NB 07).
#
# Upcoming notebooks:
# *   **`08_Hardware_DFU_Mode.py`**: How to put devices into Device Firmware Update (DFU) mode for firmware updates.
# *   **`09_Watertank_Continuous_Operation.py`**: A more complex example integrating many of these concepts for continuous watertank testing with temperature monitoring.

# End of Notebook 07
