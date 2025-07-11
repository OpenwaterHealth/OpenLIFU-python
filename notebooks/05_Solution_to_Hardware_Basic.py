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

# # 05: Sending a Basic Solution to Hardware & Triggering
#
# This notebook demonstrates how to send a defined `Solution` object to the OpenLIFU hardware and then trigger the sonication. It builds upon Notebook 03 (Solution Generation) and Notebook 04 (Hardware Connection).
#
# **Crucial Safety Note:** Running this notebook will cause the connected OpenLIFU transducer to emit ultrasound if the High Voltage (HV) power supply is enabled. Ensure the transducer is safely set up (e.g., in a water tank, appropriately coupled) before proceeding with triggering.

# ## 1. Imports

# +
import numpy as np
import time

from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.geo import Point
from openlifu.xdc import Transducer # For getting numelements
from openlifu.db import Database    # For loading a transducer
from pathlib import Path
# -

# ## 2. Create or Load a Transducer (to determine num_elements)
#
# A `Solution` needs delays and apodizations matching the number of elements in the transducer.
# We'll load a transducer to get this information.

# +
transducer = None
db_path_found = None
paths_to_check = [Path.cwd() / "db_dvc", Path.cwd() / ".." / "db_dvc"]
for path_check in paths_to_check:
    if path_check.exists() and path_check.is_dir() and (path_check / "transducers").exists():
        db_path_found = path_check.resolve()
        break

if db_path_found:
    db = Database(db_path_found)
    available_transducers = db.list_transducers()
    if available_transducers:
        trans_id_to_load = 'openlifu_2x400_evt1' if 'openlifu_2x400_evt1' in available_transducers else available_transducers[0]
        try:
            transducer = db.load_transducer(trans_id_to_load)
            print(f"Successfully loaded transducer '{transducer.id}' with {transducer.numelements()} elements.")
        except Exception as e:
            print(f"Error loading transducer '{trans_id_to_load}': {e}")
else:
    print("Database directory 'db_dvc' not found. Cannot determine num_elements accurately.")

if not transducer:
    print("Using a fallback of 64 elements for solution definition. This may not match your hardware.")
    num_elements = 64
    transducer_id_for_solution = "unknown_transducer"
else:
    num_elements = transducer.numelements()
    transducer_id_for_solution = transducer.id
# -

# ## 3. Define a Simple Solution
#
# For this example, we'll create a basic `Solution` object programmatically. In a real workflow, this might come from `protocol.calc_solution()` as shown in Notebook 03.

# +
# Basic Pulse
pulse = Pulse(frequency=400e3, duration=25e-6) # 400 kHz, 25 us duration

# Basic Sequence: 5 pulses, 100ms apart, single train
sequence = Sequence(
    pulse_interval=0.1, # 100 ms
    pulse_count=5,
    pulse_train_interval=0,
    pulse_train_count=1
)

# Basic Target (used for information in Solution, not for calculation here)
target_point = Point(position=(0,0,40), units="mm")

# Delays and Apodizations:
# For simplicity, using zeros for delays (no focusing) and ones for apodizations (all elements active).
# The shape must be (1, num_elements).
delays = np.zeros((1, num_elements))
apodizations = np.ones((1, num_elements))

# Voltage: This is a relative value. Actual output depends on hardware calibration and HV settings.
voltage = 10.0 # Example voltage value

# Create the Solution object
my_solution = Solution(
    id="basic_hardware_solution",
    name="Basic Solution for Hardware Test",
    transducer_id=transducer_id_for_solution,
    delays=delays,
    apodizations=apodizations,
    pulse=pulse,
    voltage=voltage,
    sequence=sequence,
    target=target_point,
    approved=True # Solutions usually need to be 'approved' to be sent
)

print(f"Created Solution: {my_solution.name}")
print(f"  Targeting transducer: {my_solution.transducer_id}")
print(f"  Pulse: {my_solution.pulse}")
print(f"  Sequence: {my_solution.sequence}")
print(f"  Voltage: {my_solution.voltage}")
# -

# ## 4. Connect to Hardware

# +
interface = None
try:
    print("\nInitializing LIFUInterface...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()
    print(f"  TX Connected: {tx_connected}, HV Connected: {hv_connected}")
    if not tx_connected:
        raise ConnectionError("TX device is not connected. Cannot proceed.")
    # HV connection is also important for power, will be checked before sonication.
except Exception as e:
    print(f"Error initializing LIFUInterface or devices not connected: {e}")
# -

# ## 5. CRITICAL: High Voltage (HV) Power Prerequisite
#
# To actually emit ultrasound, the High Voltage (HV) power supply on the OpenLIFU console must be:
# 1.  **Set to the desired voltage level.** This is typically done via `interface.hvcontroller.set_voltage(volts)`.
# 2.  **Turned ON.** This is done via `interface.hvcontroller.turn_hv_on()`.
#
# **Notebook 06 (`06_Hardware_Console_Controls.py`) will cover HV control in detail.**
#
# **For this notebook, you have two options:**
# *   **Option A (Manual):** Manually ensure the HV supply is already ON and set to a known, safe voltage for your transducer and setup *before* running the cells below that trigger sonication.
# *   **Option B (Programmatic - Use with Extreme Caution):** Uncomment and use the example code below to programmatically set and turn on HV.
#
# **⚠️ WARNING: Incorrect HV settings can damage the transducer or be unsafe. Always start with low voltages and ensure the transducer is properly coupled in a test environment (e.g., water tank).**

# +
# Example for Option B (Programmatic HV Control - uncomment and modify carefully if needed)
# Ensure hv_connected is True before attempting this.
# _, hv_connected = interface.is_device_connected() # Re-check if needed

# desired_hv_voltage = 15.0 # Volts - START LOW AND INCREASE CAREFULLY!

# if interface and hv_connected:
#     try:
#         print(f"\nProgrammatically setting HV to {desired_hv_voltage}V and turning ON.")
#         print("Ensure this is a safe voltage for your transducer and setup.")
#
#         # Set voltage
#         if interface.hvcontroller.set_voltage(desired_hv_voltage):
#             print(f"  HV set point configured to {interface.hvcontroller.get_voltage()} V.")
#         else:
#             print(f"  Failed to set HV voltage.")
#             raise RuntimeError("Failed to set HV voltage.")
#
#         # Turn HV ON
#         if interface.hvcontroller.turn_hv_on():
#             print(f"  HV supply turned ON. Measured voltage: {interface.hvcontroller.get_voltage_out()}V") # get_voltage_out reads actual output
#             if not interface.hvcontroller.get_hv_status(): # get_hv_status checks if it's on
#                 raise RuntimeError("HV failed to turn on despite command.")
#         else:
#             print("  Failed to turn HV ON.")
#             raise RuntimeError("Failed to turn HV ON.")
#
#         print("✅ HV supply should now be active.")
#
#     except Exception as e:
#         print(f"  ❌ Error during programmatic HV setup: {e}")
#         print("  Sonication will likely fail or do nothing.")
# else:
#     print("\nHV Controller not connected or interface not available.")
#     print("Manual HV setup is required, or connect HV controller and re-run.")

# Verify HV status before proceeding (important if manually set up)
if interface and interface.hvcontroller and interface.hvcontroller.is_connected():
    print(f"\nCurrent HV Status: {'ON' if interface.hvcontroller.get_hv_status() else 'OFF'}")
    print(f"  Set Voltage: {interface.hvcontroller.get_voltage()} V")
    print(f"  Output Voltage: {interface.hvcontroller.get_voltage_out()} V")
    if not interface.hvcontroller.get_hv_status():
        print("  ⚠️ HV is OFF. Sonication will not produce output. Please turn HV ON manually or using code above.")
elif interface:
    print("\nHV Controller not connected. Cannot verify HV status. Assuming manual setup if proceeding.")
# -

# ## 6. Send Solution to Hardware
#
# The `interface.set_solution()` method is a high-level way to send all solution parameters (delays, apodizations, pulse, sequence, voltage) to the hardware.
# Alternatively, `interface.txdevice.set_solution()` can be used to program just the TX device.
#
# *   `solution`: The `Solution` object.
# *   `profile_index`: The hardware can often store multiple solutions in "profiles". This specifies which profile to use (e.g., 1-8).
# *   `profile_increment`: Whether to auto-increment the profile index after each trigger (useful for sequences of different solutions).
# *   `trigger_mode`:
#     *   `"sequence"` (or `0`): The hardware will run the number of pulses/trains defined in the `Solution`'s `Sequence` object upon each `start_trigger()`.
#     *   `"continuous"` (or `1`): The hardware will pulse continuously according to the `pulse_interval` in the `Sequence` until `stop_trigger()` is called. The `pulse_count` in the sequence is often ignored in this mode by hardware.
# *   `turn_hv_on`: A convenience flag. If `True`, it attempts to turn on HV using the voltage from `solution.voltage`. **It's generally safer to manage HV explicitly as shown above, so we'll use `turn_hv_on=False` here.**

# +
profile_to_use = 1
trigger_mode_setting = "sequence" # or "continuous"

if interface and interface.txdevice.is_connected() and my_solution:
    print(f"\nSending solution to hardware (Profile: {profile_to_use}, Mode: {trigger_mode_setting})...")
    try:
        # Using the lower-level txdevice.set_solution for more direct control here
        # It expects the solution components directly.
        sol_dict = my_solution.to_dict() # Convert solution object to dictionary format

        # Ensure the solution dictionary has all necessary keys for txdevice.set_solution
        # This typically includes 'pulse', 'delays', 'apodizations', 'sequence', 'voltage'
        # The `voltage` here is often a *scaling factor* or *reference* for the TX board,
        # not the direct HV DAC setting. The actual acoustic output pressure also depends on
        # the main HV supply voltage set on the console.

        success = interface.txdevice.set_solution(
            pulse=sol_dict['pulse'],
            delays=sol_dict['delays'],
            apodizations=sol_dict['apodizations'],
            sequence=sol_dict['sequence'],
            voltage=sol_dict['voltage'], # This voltage is for the TX board's internal gain/amplitude scaling
            trigger_mode=trigger_mode_setting,
            profile_index=profile_to_use,
            profile_increment=False # Don't auto-increment profile for this basic example
        )

        # An alternative is the higher-level interface.set_solution():
        # success = interface.set_solution(
        #     solution=my_solution,
        #     profile_index=profile_to_use,
        #     profile_increment=False,
        #     trigger_mode=trigger_mode_setting,
        #     turn_hv_on=False # Explicitly manage HV
        # )

        if success:
            print("✅ Solution successfully sent to hardware.")
        else:
            print("❌ Failed to send solution to hardware.")
            raise RuntimeError("Failed to send solution")

    except Exception as e:
        print(f"  Error sending solution: {e}")
else:
    print("\nInterface not ready or solution not defined. Cannot send solution.")
# -

# ## 7. Triggering the Sonication
#
# Once the solution is programmed, you can start and stop the sonication.
#
# **⚠️ SAFETY WARNING: This cell will start ultrasound emission if the solution was sent successfully AND HV power is ON.**
# **Ensure your setup is safe before running.**

# +
sonication_duration_sec = 0 # Will be determined by sequence if mode is "sequence"

if interface and interface.txdevice.is_connected() and my_solution:
    if trigger_mode_setting == "sequence":
        # Calculate approximate duration for sequence mode
        s = my_solution.sequence
        one_train_duration = s.pulse_count * s.pulse_interval
        sonication_duration_sec = s.pulse_train_count * (one_train_duration + s.pulse_train_interval) - s.pulse_train_interval
        if sonication_duration_sec <=0: sonication_duration_sec = one_train_duration # single train case
        print(f"\nSonication in '{trigger_mode_setting}' mode. Expected duration: ~{sonication_duration_sec:.2f} seconds.")
    else: # continuous
        sonication_duration_sec = 2.0 # Arbitrary duration for continuous example
        print(f"\nSonication in '{trigger_mode_setting}' mode. Will run for {sonication_duration_sec:.2f} seconds in this example.")

    input(f"Press Enter to START sonication (ensure safety precautions)...")

    try:
        print("  Starting trigger...")
        if interface.txdevice.start_trigger():
            print(f"  ✅ Trigger started. Sonicating for ~{sonication_duration_sec:.2f} seconds...")
            time.sleep(sonication_duration_sec + 0.5) # Wait for sequence to complete or for continuous duration + buffer

            print("  Stopping trigger...")
            if interface.txdevice.stop_trigger():
                print("  ✅ Trigger stopped successfully.")
            else:
                print("  ❌ Failed to stop trigger. Try stopping HV power if necessary.")
        else:
            print("  ❌ Failed to start trigger.")
    except Exception as e:
        print(f"  Error during triggering: {e}")
else:
    print("\nInterface not ready. Cannot trigger sonication.")
# -

# ## 8. (Optional) Turn Off HV Power
# After sonication, it's good practice to turn off the HV supply if you previously turned it on programmatically, especially if you are done with experiments.

# +
# Example for turning HV OFF (uncomment if you turned it ON in step 5)
# _, hv_connected = interface.is_device_connected()
# if interface and hv_connected and interface.hvcontroller.get_hv_status():
#     print("\nProgrammatically turning HV OFF.")
#     try:
#         if interface.hvcontroller.turn_hv_off():
#             print("  ✅ HV supply turned OFF.")
#         else:
#             print("  ❌ Failed to turn HV OFF.")
#     except Exception as e:
#         print(f"  Error turning HV OFF: {e}")
# -

# ## 9. Cleanup

if interface:
    print("\nFinished with hardware interaction for this notebook.")
    del interface
    print("LIFUInterface instance deleted.")

# ## Next Steps
#
# This notebook covered the basics of sending a solution and triggering.
# *   **Notebook 06 (`06_Hardware_Console_Controls.py`):** For detailed control over the HV Controller (power, fans, temperature). This is essential for managing the power state required by this notebook.
# *   **Notebook 09 (`09_Watertank_Continuous_Operation.py`):** For more complex, continuous operation scenarios, often used in watertank testing, including live temperature monitoring.

# End of Notebook 05
