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

# # 08: Hardware Device Firmware Update (DFU) Mode
#
# This notebook explains how to command your OpenLIFU hardware components (HV Controller/Console and TX Device) into Device Firmware Update (DFU) mode. DFU mode is a special state that allows the device's firmware to be updated, typically using an external tool like `dfu-util` or STM32CubeProgrammer.
#
# **⚠️ WARNING: Entering DFU Mode ⚠️**
# *   When a device enters DFU mode, it will **disconnect** from the `LIFUInterface` in its normal operational mode.
# *   The device will then re-enumerate on your computer's USB bus as a "DFU device".
# *   **You will need a separate DFU programming tool to perform the actual firmware update.** This notebook only shows how to enter DFU mode.
# *   To exit DFU mode (if you don't perform an update or after an update), you typically need to **power cycle the device** (turn it off and on again).
# *   Only proceed if you intend to update firmware or understand the implications.

# ## 1. Imports

# +
import time
from openlifu.io.LIFUInterface import LIFUInterface
# -

# ## 2. Connect to Hardware
#
# Initialize `LIFUInterface`. We'll need it to send the DFU command.

# +
interface = None
try:
    print("Initializing LIFUInterface...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()
    print(f"  Initial TX Connected: {tx_connected}, Initial HV Connected: {hv_connected}")
except Exception as e:
    print(f"Error initializing LIFUInterface: {e}")
    print("Hardware might not be connected or powered.")

# -

# ## 3. HV Controller (Console) - Enter DFU Mode
#
# This section shows how to command the HV Controller (Console) into DFU mode.

# ### 3.1. Check HV Controller Connection and Ping

if interface:
    _, hv_connected = interface.is_device_connected() # Re-check
    if hv_connected:
        print("\nHV Controller is connected.")
        print("Pinging HV Controller before DFU command...")
        try:
            if interface.hvcontroller.ping():
                print("  ✅ HV Controller responded to ping.")
            else:
                print("  ❌ HV Controller did not respond to ping.")
        except Exception as e:
            print(f"  ⚠️ Error pinging HV Controller: {e}")
    else:
        print("\nHV Controller not connected. Cannot command it to DFU mode.")
else:
    print("LIFUInterface not initialized.")

# ### 3.2. Command HV Controller to DFU Mode
#
# **⚠️ Running the next cell will attempt to put your HV Controller into DFU mode. ⚠️**
# *   It will disconnect from `LIFUInterface`.
# *   You will need a DFU tool (e.g., STM32CubeProgrammer) to update its firmware.
# *   Power cycle the HV Controller to return to normal operation if no update is performed.
#
# **Only uncomment and run the cell if you intend to do this.**

# +
# ##############################################################################
# # CELL TO PUT HV CONTROLLER (CONSOLE) INTO DFU MODE - RUN WITH CAUTION
# ##############################################################################
#
# if interface and interface.hvcontroller and interface.hvcontroller.is_connected():
#     print("\nAttempting to command HV Controller to DFU mode...")
#     try:
#         interface.hvcontroller.enter_dfu()
#         print("  ✅ DFU command sent to HV Controller.")
#         print("     The HV Controller should now reboot into DFU mode.")
#         print("     It will disconnect from this interface and appear as a DFU device to your OS.")
#         print("     LIFUInterface will likely lose connection to it now.")
#         # At this point, the hvcontroller object is likely no longer valid for normal operations.
#     except Exception as e:
#         print(f"  ⚠️ Error sending DFU command to HV Controller: {e}")
# else:
#     print("\nHV Controller not connected or interface not ready. Cannot send DFU command.")
#
# ##############################################################################
# print("\nIf DFU command was sent, check your OS for a DFU device.")
# print("Use STM32CubeProgrammer or dfu-util for firmware update.")
# print("Power cycle the console to return to normal operation if no update is performed.")
# -

# ## 4. TX Device - Enter DFU Mode
#
# This section shows how to command the TX Device into DFU mode.

# ### 4.1. Check TX Device Connection and Ping
# The TX device might need 12V power from the HV Controller.

# +
if interface:
    tx_connected, hv_connected = interface.is_device_connected() # Re-check
    print(f"\nInitial TX connection status for DFU section: {tx_connected}")

    if not tx_connected and hv_connected and interface.hvcontroller.is_connected():
        print("TX device not connected. Attempting to turn on 12V via HV Controller...")
        try:
            if interface.hvcontroller.turn_12v_on():
                print("  12V supply turned ON. Waiting for TX device to enumerate...")
                time.sleep(3) # Give time for TX to power up and USB to enumerate

                # It's often best to re-initialize the interface to discover newly powered devices
                print("  Re-initializing LIFUInterface to detect TX device...")
                interface.stop_monitoring() # Clean up old instance if it was monitoring
                del interface
                interface = LIFUInterface()
                tx_connected, hv_connected = interface.is_device_connected()
                print(f"  New TX Connected: {tx_connected}, New HV Connected: {hv_connected}")
            else:
                print("  Failed to turn on 12V power via HV Controller.")
        except Exception as e:
            print(f"  Error during 12V power-on sequence for TX: {e}")


    if tx_connected and interface.txdevice and interface.txdevice.is_connected():
        print("\nTX Device is connected.")
        print("Pinging TX Device before DFU command...")
        try:
            if interface.txdevice.ping():
                print("  ✅ TX Device responded to ping.")
            else:
                print("  ❌ TX Device did not respond to ping.")
        except Exception as e:
            print(f"  ⚠️ Error pinging TX Device: {e}")
    else:
        print("\nTX Device not connected, even after 12V power attempt (if applicable).")
        print("Cannot command TX Device to DFU mode.")
else:
    print("LIFUInterface not initialized.")
# -

# ### 4.2. Command TX Device to DFU Mode
#
# **⚠️ Running the next cell will attempt to put your TX Device into DFU mode. ⚠️**
# *   It will disconnect from `LIFUInterface`.
# *   You will need a DFU tool (e.g., `dfu-util`) to update its firmware.
# *   Power cycle the TX Device (or the 12V supply from console, then the console) to return to normal operation.
#
# **Only uncomment and run the cell if you intend to do this.**

# +
# ##############################################################################
# # CELL TO PUT TX DEVICE INTO DFU MODE - RUN WITH CAUTION
# ##############################################################################
#
# if interface and interface.txdevice and interface.txdevice.is_connected():
#     print("\nAttempting to command TX Device to DFU mode...")
#     try:
#         if interface.txdevice.enter_dfu(): # Returns True on successful command send
#             print("  ✅ DFU command sent successfully to TX Device.")
#             print("     The TX Device should now reboot into DFU mode.")
#             print("     It will disconnect from this interface and appear as a DFU device to your OS.")
#         else:
#             print("  ❌ DFU command to TX Device was not acknowledged or failed.")
#
#     except Exception as e:
#         print(f"  ⚠️ Error sending DFU command to TX Device: {e}")
# else:
#     print("\nTX Device not connected or interface not ready. Cannot send DFU command.")
#
# ##############################################################################
# print("\nIf DFU command was sent, check your OS for a DFU device.")
# print("Use dfu-util or other appropriate tool for firmware update on the TX device.")
# print("Power cycle the TX device (or 12V supply) to return to normal operation if no update is performed.")
# -

# ## 5. Exiting DFU Mode
#
# As mentioned, DFU mode is typically exited by a **power cycle** of the device:
# 1.  Turn off the power to the device.
# 2.  Wait a few seconds.
# 3.  Turn the power back on.
#
# The device should then boot into its normal operational firmware. If you have just flashed new firmware, this will be the first boot of that new firmware.

# ## 6. Cleanup
#
# If you have sent a DFU command, the `interface` object may no longer have valid connections.

if interface:
    print("\nDFU operations notebook finished.")
    print("If devices were put into DFU mode, they are no longer connected to this interface.")
    # interface.stop_monitoring() # Good practice if async monitoring was ever started
    del interface
    print("LIFUInterface instance deleted (or was already invalid if DFU occurred).")

# ## Next Steps
#
# This notebook showed how to command devices into DFU mode, a prerequisite for firmware updates.
#
# *   **`09_Watertank_Continuous_Operation.py`**: A more applied example demonstrating continuous pulsing for watertank testing, including setting up solutions and live temperature monitoring. This combines many concepts from previous notebooks.

# End of Notebook 08
