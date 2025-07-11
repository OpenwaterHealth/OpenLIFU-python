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

# # 06: Hardware Console (HV Controller) Controls
#
# This notebook provides a comprehensive guide to interacting with the OpenLIFU HV Controller (often referred to as the "console"). The HV Controller is responsible for power management (12V for the system and High Voltage for sonication), temperature monitoring, fan control, and status LEDs.
#
# **⚠️ SAFETY FIRST! This notebook involves controlling High Voltage (HV). Incorrect operation can be dangerous and may damage your hardware. Always:**
# *   **Understand what each command does before running it.**
# *   **Start with low HV settings and gradually increase if necessary.**
# *   **Ensure the transducer is handled safely, especially when HV is active.**
# *   **Monitor system temperatures, especially during prolonged operation.**

# ## 1. Imports

# +
import time
import base58 # For decoding hardware IDs from the console
from openlifu.io.LIFUInterface import LIFUInterface
# -

# ## 2. Connect to Hardware and Verify HV Controller
#
# We start by initializing `LIFUInterface` and ensuring the HV Controller is connected.

# +
interface = None
hv_controller = None # Will assign interface.hvcontroller here for convenience

try:
    print("Initializing LIFUInterface...")
    interface = LIFUInterface(TX_test_mode=True) # TX_test_mode=True can sometimes help if only focusing on console
    tx_connected, hv_connected = interface.is_device_connected()
    print(f"  TX Connected: {tx_connected}, HV Connected: {hv_connected}")

    if not hv_connected:
        raise ConnectionError("HV Controller (Console) is not connected. This notebook requires it.")
    hv_controller = interface.hvcontroller
    print("✅ HV Controller is connected and accessible via `hv_controller`.")

except Exception as e:
    print(f"Error initializing LIFUInterface or HV Controller not connected: {e}")
# -

# ## 3. Basic Information

if hv_controller:
    # Ping
    print("\nPinging HV Controller...")
    try:
        if hv_controller.ping():
            print("  ✅ HV Controller responded to ping.")
        else:
            print("  ❌ HV Controller did NOT respond or an error occurred.")
    except Exception as e:
        print(f"  ⚠️ Error pinging HV Controller: {e}")

    # Get Version
    print("\nGetting HV Controller Firmware Version...")
    try:
        version_hv = hv_controller.get_version()
        print(f"  HV Controller Firmware Version: {version_hv}")
    except Exception as e:
        print(f"  ⚠️ Error getting firmware version: {e}")

    # Get Hardware ID
    print("\nGetting HV Controller Hardware ID...")
    try:
        hw_id_hv = hv_controller.get_hardware_id()
        print(f"  Raw Hardware ID: {hw_id_hv}")
        if hw_id_hv and isinstance(hw_id_hv, str) and len(hw_id_hv) == 16: # Expected format
            encoded_id = base58.b58encode(bytes.fromhex(hw_id_hv)).decode()
            print(f"  Decoded Full ID (example format): OW-LIFU-CON-{encoded_id}")
        else:
            print(f"  Hardware ID format unexpected for Base58 encoding: {hw_id_hv}")
    except Exception as e:
        print(f"  ⚠️ Error getting hardware ID: {e}")

# ## 4. LED Control

if hv_controller:
    # Toggle basic status LED (often a green LED on the board)
    print("\nToggling basic status LED...")
    try:
        # The effect of toggle_led might be subtle or control a specific diagnostic LED
        # It's often a quick way to see if basic commands are working
        current_led_state_unknown = hv_controller.toggle_led() # May not return a meaningful state
        print("  toggle_led() command sent. Observe device for LED change.")
        # Toggling it back to try and restore original state, though its actual state is not read here.
        # time.sleep(0.5)
        # hv_controller.toggle_led()
    except Exception as e:
        print(f"  ⚠️ Error toggling LED: {e}")

    # RGB LED Control (if equipped)
    # The RGB LED often indicates system status (e.g., idle, running, error)
    print("\nControlling RGB LED...")
    try:
        initial_rgb_state = hv_controller.get_rgb_led()
        print(f"  Initial RGB LED state: {initial_rgb_state}")

        print("  Setting RGB LED to state 2 (example: blue or other color)...")
        if hv_controller.set_rgb_led(rgb_state=2): # State numbers are device-specific
            print(f"  RGB LED state set. New state: {hv_controller.get_rgb_led()}")
        else:
            print("  Failed to set RGB LED state.")

        time.sleep(1)
        print(f"  Restoring initial RGB LED state ({initial_rgb_state})...")
        hv_controller.set_rgb_led(rgb_state=initial_rgb_state)
        print(f"  RGB LED state restored. Current state: {hv_controller.get_rgb_led()}")

    except Exception as e:
        print(f"  ⚠️ Error controlling RGB LED: {e}")
        print("     (This console might not support RGB LED control or the command failed).")

# ## 5. Temperature Monitoring

if hv_controller:
    print("\nReading Temperatures from Console Sensors...")
    try:
        temp1 = hv_controller.get_temperature1()
        print(f"  Temperature Sensor 1: {temp1}°C")
    except Exception as e:
        print(f"  ⚠️ Error reading Temperature Sensor 1: {e}")

    try:
        temp2 = hv_controller.get_temperature2() # May not be present on all consoles
        print(f"  Temperature Sensor 2: {temp2}°C")
    except Exception as e:
        print(f"  ⚠️ Error reading Temperature Sensor 2: {e} (Sensor might not be available).")

# ## 6. Fan Control
# The console may have one or more fans. Fan IDs are usually 0-indexed.

if hv_controller:
    print("\nControlling Fans...")
    # Example for Fan 0 (e.g., Bottom Fan)
    fan_id_0 = 0
    try:
        initial_fan0_speed = hv_controller.get_fan_speed(fan_id=fan_id_0)
        print(f"  Initial Fan {fan_id_0} speed: {initial_fan0_speed}%")

        print(f"  Setting Fan {fan_id_0} speed to 25%...")
        if hv_controller.set_fan_speed(fan_id=fan_id_0, fan_speed=25):
             print(f"  Fan {fan_id_0} speed set. New speed: {hv_controller.get_fan_speed(fan_id=fan_id_0)}%")
        else:
            print(f"  Failed to set Fan {fan_id_0} speed.")
        time.sleep(1)
        print(f"  Restoring Fan {fan_id_0} to initial speed ({initial_fan0_speed}%)...")
        hv_controller.set_fan_speed(fan_id=fan_id_0, fan_speed=initial_fan0_speed)
        print(f"  Fan {fan_id_0} speed restored. Current speed: {hv_controller.get_fan_speed(fan_id=fan_id_0)}%")
    except Exception as e:
        print(f"  ⚠️ Error controlling Fan {fan_id_0}: {e} (Fan might not be present or command failed).")

    # Example for Fan 1 (e.g., Top Fan) - might not exist on all consoles
    fan_id_1 = 1
    try:
        initial_fan1_speed = hv_controller.get_fan_speed(fan_id=fan_id_1)
        print(f"\n  Initial Fan {fan_id_1} speed: {initial_fan1_speed}%")
        # ... (similar set and restore logic as for fan_id_0) ...
    except Exception as e:
        print(f"\n  Note: Could not get initial speed for Fan {fan_id_1}, it might not be present: {e}")


# ## 7. 12V Power Supply Control
# The 12V supply is crucial as it often powers the TX (Transmitter) board and other system components.

if hv_controller:
    print("\nControlling 12V Power Supply...")
    try:
        # Check initial status
        is_12v_on_initially = hv_controller.get_12v_status()
        print(f"  Initial 12V status: {'ON' if is_12v_on_initially else 'OFF'}")

        if not is_12v_on_initially:
            print("  Turning 12V ON...")
            if hv_controller.turn_12v_on():
                print("  ✅ 12V supply turned ON.")
                print(f"  Current 12V status: {'ON' if hv_controller.get_12v_status() else 'OFF'}")
                print("  (If TX board is powered by this, it should now be booting up.)")
                print("  (Allow a few seconds for TX to boot if you need to use it immediately.)")
                # interface.check_tx_connection() # Hypothetical method to re-check TX
                time.sleep(3) # Wait for TX to potentially boot
                # You might want to re-check interface.is_device_connected() here
                # tx_conn_after_12v, _ = interface.is_device_connected()
                # print(f"  TX Connection status after 12V ON and wait: {tx_conn_after_12v}")
            else:
                print("  ❌ Failed to turn 12V ON.")
        else:
            print("  12V is already ON.")

        # Example: Turning 12V OFF (if it was turned on by this cell or was on initially)
        # Be cautious: Turning off 12V will power down the TX board.
        # For this example, we'll turn it off if it was initially off.
        if not is_12v_on_initially and hv_controller.get_12v_status(): # If we turned it on
            print("\n  Turning 12V OFF (as it was initially off)...")
            if hv_controller.turn_12v_off():
                print("  ✅ 12V supply turned OFF.")
            else:
                print("  ❌ Failed to turn 12V OFF.")
        elif is_12v_on_initially:
             print("\n  12V was initially ON, leaving it ON for this example.")


    except Exception as e:
        print(f"  ⚠️ Error controlling 12V power: {e}")

# ## 8. High Voltage (HV) Power Control
#
# **⚠️ EXTREME CAUTION: HIGH VOLTAGE! ⚠️**
# *   Ensure your transducer is safely set up (e.g., in a water tank, properly coupled).
# *   **ALWAYS START WITH THE LOWEST POSSIBLE VOLTAGE AND INCREASE GRADUALLY AND CAREFULLY.**
# *   Incorrect HV settings can be dangerous and can permanently damage your transducer or electronics.
# *   The `voltage` set here is the target for the HV power supply rails (e.g., setting 20V means +/-20V).

if hv_controller:
    print("\n--- High Voltage (HV) Power Control ---")
    print("⚠️ ENSURE ALL SAFETY PRECAUTIONS ARE IN PLACE BEFORE PROCEEDING! ⚠️")

    # Define a SAFE, LOW starting voltage for this test.
    test_hv_voltage = 5.0  # Volts. START EXTREMELY LOW (e.g., 5V or 10V).
    print(f"\nSetting HV target to a LOW, SAFE test voltage: {test_hv_voltage} V")

    try:
        # 1. Set HV Voltage Level
        print(f"\n  Attempting to set HV voltage to {test_hv_voltage} V...")
        if hv_controller.set_voltage(voltage=test_hv_voltage):
            current_set_voltage = hv_controller.get_voltage()
            print(f"  ✅ HV voltage set point configured to: {current_set_voltage} V")
            if abs(current_set_voltage - test_hv_voltage) > 0.5: # Check if it set correctly
                 print(f"  ⚠️ Warning: Set voltage {current_set_voltage}V differs from target {test_hv_voltage}V.")
        else:
            print(f"  ❌ Failed to set HV voltage to {test_hv_voltage} V.")
            raise RuntimeError(f"Failed to set HV to {test_hv_voltage}V") # Stop before turning on

        # 2. Turn HV Supply ON
        # Ensure it's off before trying to turn on, for a clean test
        if hv_controller.get_hv_status():
            print("\n  HV is already ON. Turning OFF first for a clean test...")
            hv_controller.turn_hv_off()
            time.sleep(0.5)
            if hv_controller.get_hv_status():
                raise RuntimeError("Failed to turn HV OFF before test.")

        print("\n  Attempting to turn HV supply ON...")
        if hv_controller.turn_hv_on():
            print("  ✅ HV supply turned ON.")
            time.sleep(1) # Allow time for voltage to stabilize and be measured
            hv_on_status = hv_controller.get_hv_status()
            actual_output_voltage = hv_controller.get_voltage_out()
            print(f"  HV Status: {'ON' if hv_on_status else 'OFF'}")
            print(f"  Actual Measured Output Voltage: {actual_output_voltage} V")
            if not hv_on_status:
                print("  ❌ HV failed to stay ON or report as ON.")
        else:
            print("  ❌ Failed to turn HV supply ON.")

        # 3. Turn HV Supply OFF (CRITICAL for safety after testing)
        if hv_controller.get_hv_status(): # Only if it's on
            print("\n  Attempting to turn HV supply OFF...")
            if hv_controller.turn_hv_off():
                print("  ✅ HV supply turned OFF.")
                time.sleep(0.5)
                hv_on_status_after_off = hv_controller.get_hv_status()
                actual_output_voltage_after_off = hv_controller.get_voltage_out()
                print(f"  HV Status: {'ON' if hv_on_status_after_off else 'OFF'}")
                print(f"  Actual Measured Output Voltage: {actual_output_voltage_after_off} V")
                if hv_on_status_after_off:
                     print("  ❌ HV failed to turn OFF or report as OFF.")
            else:
                print("  ❌ Failed to turn HV supply OFF via command.")
        else:
            print("\n  HV was not ON, no need to turn off again.")

    except Exception as e:
        print(f"  ⚠️ An error occurred during HV control: {e}")
        print("     For safety, attempting to turn HV OFF if possible...")
        if hv_controller and hv_controller.is_connected():
            try:
                hv_controller.turn_hv_off()
                print("     Attempted safety HV OFF successful.")
            except Exception as e_off:
                print(f"     Could not perform safety HV OFF: {e_off}")

# ## 9. Echo Command
# The echo command is a simple way to test data transmission to and from the device.

if hv_controller:
    print("\nTesting Echo Command...")
    echo_test_data = b'Hello OpenLIFU Console!'
    try:
        response_data, response_len = hv_controller.echo(echo_data=echo_test_data)
        if response_len > 0 and response_data == echo_test_data:
            print(f"  ✅ Echo successful! Sent: '{echo_test_data.decode()}', Received: '{response_data.decode()}'")
        elif response_len > 0:
            print(f"  ⚠️ Echo mismatch. Sent: '{echo_test_data.decode()}', Received: '{response_data.decode()}'")
        else:
            print("  ❌ Echo failed. No data or zero length data returned.")
    except Exception as e:
        print(f"  ⚠️ Error during echo test: {e}")

# ## 10. Soft Reset
# The console can be reset via a software command. This is similar to a reboot.
# After a soft reset, you'll typically need to re-initialize the `LIFUInterface` or at least re-check connections.

if hv_controller:
    print("\n--- Soft Reset ---")
    print("The following cell, if uncommented and run, will reset the HV Controller.")
    print("You would typically need to re-initialize LIFUInterface after a reset.")

    # print("If you wish to test soft_reset, uncomment the lines below.")
    # print("Consider this an ADVANCED operation for debugging.")
    # try:
    #     print("  Attempting to soft reset HV Controller...")
    #     # Note: After soft_reset, the hv_controller object and interface might become invalid.
    #     if hv_controller.soft_reset():
    #         print("  ✅ Soft reset command sent successfully.")
    #         print("     The HV Controller is now rebooting.")
    #         print("     You will likely need to re-run previous cells to reconnect.")
    #         hv_controller = None # Invalidate object
    #         interface = None     # Invalidate interface
    #     else:
    #         print("  ❌ Soft reset command failed or was not acknowledged.")
    # except Exception as e:
    #     print(f"  ⚠️ Error during soft reset: {e} (Communication might be lost)")
    #     hv_controller = None
    #     interface = None

# ## 11. Cleanup
# It's good practice to release the interface when done.

if interface:
    print("\nFinished with HV Controller interactions for this notebook.")
    # If HV was turned on and not turned off in section 8, consider turning it off now for safety:
    # if hv_controller and hv_controller.is_connected() and hv_controller.get_hv_status():
    #    print("Ensuring HV is OFF before exiting...")
    #    hv_controller.turn_hv_off()
    del interface
    interface = None
    hv_controller = None
    print("LIFUInterface instance deleted.")

# ## Next Steps
#
# This notebook covered detailed control of the HV Controller (Console).
# Key takeaways:
# *   Management of 12V power (crucial for TX board).
# *   Safe and controlled operation of High Voltage.
# *   Monitoring temperatures and controlling fans.
#
# Next notebooks will cover:
# *   **`07_Hardware_Transmitter_Controls.py`**: Interacting with the TX (Transmitter) board.
# *   **`08_Hardware_DFU_Mode.py`**: Device Firmware Update mode.
# *   **`09_Watertank_Continuous_Operation.py`**: A more complex example combining solution generation, hardware control, and continuous monitoring for watertank tests.

# End of Notebook 06
