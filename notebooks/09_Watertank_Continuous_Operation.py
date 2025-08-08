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

# # 09: Watertank Continuous Operation with Temperature Monitoring
#
# This notebook provides a comprehensive example for running continuous or long-duration sonication experiments, typically in a watertank setup. It features:
# *   Configuration of solution parameters.
# *   Hardware setup and connection.
# *   Programming the OpenLIFU device for continuous ultrasound emission.
# *   Real-time temperature monitoring (Console, TX, Ambient) in a separate thread.
# *   Logging temperature data to a CSV file.
# *   Automated safety shutdowns based on user-defined temperature limits and rapid temperature increases.
# *   Overall experiment timeout for safety.
#
# **⚠️ EXTREME CAUTION: This notebook operates hardware that emits ultrasound and controls High Voltage. ⚠️**
# *   **Thoroughly understand each part of this notebook before execution.**
# *   **Ensure your experimental setup (watertank, transducer coupling, targets) is safe and appropriate.**
# *   **Always start with low power/voltage settings and short durations.**
# *   **Monitor the system closely during operation.**
# *   **Incorrect settings or misuse can be dangerous and may damage hardware.**

# ## 1. Imports

# +
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

# For notebook interaction (optional, for a "Stop" button like experience)
# from IPython.display import display, Markdown
# import ipywidgets as widgets
# -

# ## 2. Configuration Parameters
# Adjust these parameters for your specific experiment.

# +
# %%% Configuration Cell

# --- Target and Acoustic Parameters ---
target_x_mm = 0.0
target_y_mm = 0.0
target_z_mm = 50.0  # Focal depth in mm

frequency_kHz = 400.0   # Ultrasound frequency in kHz
solution_voltage = 10.0 # Voltage for the Solution object (amplitude scaling factor for TX)
                          # Actual acoustic output also depends on HV Console setting.
pulse_duration_msec = 5.0   # Duration of each ultrasound pulse in milliseconds
pulse_interval_msec = 100.0 # Interval between start of pulses (PRI) in milliseconds

# --- Hardware Configuration ---
num_modules = 2  # Number of TX modules (e.g., 2 for a 2x transducer like openlifu_2x400_evt1)
                 # This helps select the correct transducer from the database.
use_external_power_supply = False # If True, console HV and 12V are NOT controlled by this script.
                                 # User must manage external HV and 12V supplies.
                                 # If False, script will attempt to control console HV and 12V.

# --- Temperature Safety Limits (Celsius) ---
console_shutoff_temp_C = 65.0  # Max console temperature
tx_shutoff_temp_C = 65.0       # Max TX device temperature
ambient_shutoff_temp_C = 60.0    # Max ambient temperature (on TX board)

# --- Rapid Temperature Increase Safety Limits ---
# These values define how quickly temperatures can rise before triggering a shutdown.
# If temp rises by more than 'rapid_temp_increase_per_second_shutoff_C' within 'log_interval' seconds.
rapid_temp_increase_per_second_shutoff_C = 3.0 # Degrees C per log_interval
# If temp exceeds 'rapid_temp_shutoff_C' within 'rapid_temp_shutoff_seconds' of starting sonication.
rapid_temp_shutoff_C = 40.0    # Absolute temperature limit for initial rapid rise
rapid_temp_shutoff_seconds = 10 # Time window for initial rapid rise check

# --- Logging and Timeout ---
log_interval_sec = 1.0  # How often to log temperatures (seconds)
log_to_csv_file = True  # Enable/disable CSV logging of temperatures
experiment_timeout_minutes = 30 # Overall safety timeout for the experiment (minutes)

# --- Derived Parameters (do not change these directly) ---
# peak_to_peak_voltage = solution_voltage * 2 # This was in original, but solution_voltage is for TX scaling
# Actual HV setting is separate if not using external supply.
# -

# ## 3. Logging Setup

# +
logger = logging.getLogger("WatertankExperiment")
logger.setLevel(logging.INFO)
logger.handlers.clear() # Clear existing handlers to avoid duplicates if re-running cell

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"))
logger.addHandler(console_handler)
logger.propagate = False # Prevent passing to root logger
# -

# ## 4. Paths and Database Setup

# +
# Assuming this notebook is in a 'notebooks' subdirectory of the main project
# and 'db_dvc' and 'logs' are at the project root. Adjust if your structure differs.
notebook_dir = Path.cwd()
project_root = notebook_dir.parent # Assumes notebooks/ is one level down from project root

db_path = project_root / "db_dvc"
log_output_path = project_root / "logs"

if log_to_csv_file:
    log_output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temperature CSV logs will be saved to: {log_output_path}")

try:
    lifu_db = Database(db_path)
    logger.info(f"Successfully connected to OpenLIFU database at: {db_path}")
    # Example: list transducers to verify
    # logger.debug(f"Available transducers: {lifu_db.list_transducers()}")
except Exception as e:
    logger.exception(f"Failed to connect to OpenLIFU database at {db_path}: {e}")
    lifu_db = None
# -

# ## 5. Transducer Loading and Solution Definition

# +
transducer = None
solution = None

if lifu_db:
    try:
        # Construct transducer ID based on num_modules, assuming a naming convention
        # e.g., "openlifu_2x400_evt1" for num_modules=2
        # This part may need adjustment based on actual transducer IDs in your database
        transducer_id = f"openlifu_{num_modules}x{int(frequency_kHz)}_evt1" # Example
        logger.info(f"Attempting to load transducer: {transducer_id}")
        transducer = lifu_db.load_transducer(transducer_id)
        transducer.sort_by_pin() # Sort elements by pin number for hardware compatibility
        logger.info(f"Successfully loaded transducer: {transducer.name} with {transducer.numelements()} elements.")

        # Define Target
        target = Point(position=(target_x_mm, target_y_mm, target_z_mm), units="mm")
        logger.info(f"Target defined at: {target.position_str}")

        # Calculate Delays and Apodizations (Simple geometric focusing)
        focus_mm = target.get_position(units="mm")
        elem_pos_mm = transducer.get_positions(units="mm")
        distances_m = np.sqrt(np.sum((focus_mm - elem_pos_mm)**2, 1)).reshape(1, -1) * 1e-3
        speed_of_sound_mps = 1500.0 # m/s in water
        time_of_flight_s = distances_m / speed_of_sound_mps
        delays_s = time_of_flight_s.max() - time_of_flight_s

        apodizations = np.ones((1, transducer.numelements()))
        # Example: De-select elements if needed for specific tests
        # apodizations[:, some_indices] = 0
        logger.info(f"Delays and apodizations calculated for {transducer.numelements()} elements.")
        logger.debug(f"Example Delays (first 5): {delays_s[0, :5]}")

        # Create Pulse object
        pulse = Pulse(frequency=frequency_kHz * 1e3, duration=pulse_duration_msec * 1e-3)
        logger.info(f"Pulse defined: {pulse}")

        # Create Sequence object
        # For continuous mode, pulse_count might be set very high or to a special value.
        # The hardware interprets trigger_mode="continuous" to run indefinitely until stop_trigger.
        # A nominal pulse_count is still needed for the Sequence object.
        # Let's make it equivalent to 1 minute of pulsing for sequence definition.
        num_pulses_in_one_minute = int(60.0 / (pulse_interval_msec * 1e-3))
        sequence = Sequence(
            pulse_interval=pulse_interval_msec * 1e-3,
            pulse_count=num_pulses_in_one_minute, # This is somewhat nominal if trigger_mode is continuous
            pulse_train_interval=0,      # Only one "train" for continuous mode
            pulse_train_count=1          # Only one "train"
        )
        logger.info(f"Sequence defined: {sequence}")

        # Create Solution object
        solution = Solution(
            delays=delays_s,
            apodizations=apodizations,
            pulse=pulse,
            voltage=solution_voltage, # This is the scaling factor for TX board amplitude
            sequence=sequence,
            transducer_id=transducer.id,
            target=target,
            name="WatertankContinuousSolution",
            approved=True
        )
        logger.info(f"Solution object created: {solution.name}")

    except Exception as e:
        logger.exception(f"Error during transducer loading or solution definition: {e}")
        transducer = None # Ensure it's None if setup fails
        solution = None

else:
    logger.exception("Database not available. Cannot proceed with transducer/solution setup.")

# These will be used by the threads and main control logic
# Global variables to hold interface and thread objects
lifu_interface_global = None
temperature_thread_global = None
stop_logging_event_global = threading.Event()
operation_shutdown_event_global = threading.Event() # Signaled by temp logger on error or by user
# -

# ## 6. Temperature Logging Thread Function
# This function runs in a separate thread to monitor and log temperatures, and trigger safety shutdowns.

def temperature_logging_thread_func(
    interface: LIFUInterface,
    stop_event: threading.Event,
    shutdown_event: threading.Event,
    csv_filepath: Path | None,
    params: dict
):
    """
    Monitors temperatures, logs to CSV, and enforces safety shutdowns.
    Args:
        interface: The LIFUInterface instance.
        stop_event: Event to signal this thread to stop logging and exit.
        shutdown_event: Event that this thread will set if a safety condition is met.
        csv_filepath: Path to the CSV file for logging. None to disable.
        params: Dictionary of configuration parameters.
    """
    thread_name = threading.current_thread().name
    logger.info(f"Temperature logging thread '{thread_name}' started.")

    start_time_ns = time.monotonic_ns() # For precise timing of rapid increase checks

    # Previous temperature readings for calculating rate of change
    prev_tx_temp = None
    prev_ambient_temp = None
    prev_console_temp = None # Only if not using external power supply

    if not csv_filepath:
        logger.info("CSV logging disabled.")
        return

    with open(csv_filepath, "w") as csv_file:
        header = ("Timestamp (ISO),Elapsed Time (s),"
                  "Frequency (kHz),Pulse Duration (ms),Pulse Interval (ms),Solution Voltage (V),"
                  "Console Temp (C),TX Temp (C),Ambient Temp (C),HV Setpoint (V), HV Output (V), Status\n")
        csv_file.write(header)
        logger.info(f"Logging temperatures to CSV: {csv_filepath}")

        loop_count = 0
        while not (stop_event.is_set() or shutdown_event.is_set()):
            loop_start_ns = time.monotonic_ns()
            current_timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
            elapsed_seconds = (loop_start_ns - start_time_ns) / 1e9

            # --- Read Temperatures ---
            tx_temp_C, ambient_temp_C, console_temp_C = "N/A", "N/A", "N/A"
            hv_set_V, hv_out_V = "N/A", "N/A"
            status_message = "OK"

            try:
                tx_temp_C = interface.txdevice.get_temperature()
                ambient_temp_C = interface.txdevice.get_ambient_temperature()
                if not params['use_external_power_supply'] and interface.hvcontroller.is_connected():
                    console_temp_C = interface.hvcontroller.get_temperature1()
                    hv_set_V = interface.hvcontroller.get_voltage()
                    hv_out_V = interface.hvcontroller.get_voltage_out()
            except Exception as e:
                logger.warning(f"Failed to read some temperatures: {e}")
                status_message = "TEMP_READ_ERROR"

            # --- Safety Checks ---
            # Check for initial rapid temperature increase (within first few seconds)
            within_initial_time_threshold = elapsed_seconds < params['rapid_temp_shutoff_seconds']

            # Check for too high of a temperature increase between readings
            if prev_console_temp is not None and console_temp_C != "N/A" and not params['use_external_power_supply'] and (console_temp_C - prev_console_temp) > params['rapid_temp_increase_per_second_shutoff_C']:
                    msg = f"Console temp rapidly increased: {prev_console_temp:.1f}C to {console_temp_C:.1f}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_CONSOLE_TEMP_RISE;{msg}"
                    shutdown_event.set()
            if console_temp_C != "N/A":
                prev_console_temp = console_temp_C
            if prev_tx_temp is not None and tx_temp_C != "N/A" and (tx_temp_C - prev_tx_temp) > params['rapid_temp_increase_per_second_shutoff_C']:
                    msg = f"TX temp rapidly increased: {prev_tx_temp:.1f}C to {tx_temp_C:.1f}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_TX_TEMP_RISE;{msg}"
                    shutdown_event.set()
            if tx_temp_C != "N/A":
                prev_tx_temp = tx_temp_C

            if prev_ambient_temp is not None and ambient_temp_C != "N/A" and (ambient_temp_C - prev_ambient_temp) > params['rapid_temp_increase_per_second_shutoff_C']:
                    msg = f"Ambient temp rapidly increased: {prev_ambient_temp:.1f}C to {ambient_temp_C:.1f}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_AMBIENT_TEMP_RISE;{msg}"
                    shutdown_event.set()
            if ambient_temp_C != "N/A":
                prev_ambient_temp = ambient_temp_C


            # Initial period rapid rise absolute limit
            if within_initial_time_threshold:
                if console_temp_C != "N/A" and not params['use_external_power_supply'] and console_temp_C > params['rapid_temp_shutoff_C']:
                    msg = f"Console temp {console_temp_C:.1f}C > initial rapid limit {params['rapid_temp_shutoff_C']}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_CONSOLE_INITIAL_LIMIT;{msg}"
                    shutdown_event.set()
                if tx_temp_C != "N/A" and tx_temp_C > params['rapid_temp_shutoff_C']:
                    msg = f"TX temp {tx_temp_C:.1f}C > initial rapid limit {params['rapid_temp_shutoff_C']}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_TX_INITIAL_LIMIT;{msg}"
                    shutdown_event.set()
                if ambient_temp_C != "N/A" and ambient_temp_C > params['rapid_temp_shutoff_C']:
                    msg = f"Ambient temp {ambient_temp_C:.1f}C > initial rapid limit {params['rapid_temp_shutoff_C']}C"
                    logger.critical(f"SHUTDOWN: {msg}")
                    status_message = f"CRITICAL_AMBIENT_INITIAL_LIMIT;{msg}"
                    shutdown_event.set()

            # Absolute temperature limits
            if console_temp_C != "N/A" and not params['use_external_power_supply'] and console_temp_C > params['console_shutoff_temp_C']:
                msg = f"Console temp {console_temp_C:.1f}C > limit {params['console_shutoff_temp_C']}C"
                logger.critical(f"SHUTDOWN: {msg}")
                status_message = f"CRITICAL_CONSOLE_ABS_LIMIT;{msg}"
                shutdown_event.set()
            if tx_temp_C != "N/A" and tx_temp_C > params['tx_shutoff_temp_C']:
                msg = f"TX temp {tx_temp_C:.1f}C > limit {params['tx_shutoff_temp_C']}C"
                logger.critical(f"SHUTDOWN: {msg}")
                status_message = f"CRITICAL_TX_ABS_LIMIT;{msg}"
                shutdown_event.set()
            if ambient_temp_C != "N/A" and ambient_temp_C > params['ambient_shutoff_temp_C']:
                msg = f"Ambient temp {ambient_temp_C:.1f}C > limit {params['ambient_shutoff_temp_C']}C"
                logger.critical(f"SHUTDOWN: {msg}")
                status_message = f"CRITICAL_AMBIENT_ABS_LIMIT;{msg}"
                shutdown_event.set()

            # Log to console
            log_msg_console = (f"Elapsed: {elapsed_seconds:6.1f}s | "
                               f"Console: {console_temp_C}°C | TX: {tx_temp_C}°C | Ambient: {ambient_temp_C}°C | "
                               f"HV: {hv_set_V}V (Set), {hv_out_V}V (Out) | Status: {status_message}")
            if shutdown_event.is_set():
                logger.critical(log_msg_console)
            elif loop_count % 5 == 0 :
                logger.info(log_msg_console) # Log to console less frequently
            else:
                logger.debug(log_msg_console)


            # Log to CSV
            if csv_file:
                try:
                    csv_line = (f"{current_timestamp_iso},{elapsed_seconds:.3f},"
                                f"{params['frequency_kHz']},{params['pulse_duration_msec']},{params['pulse_interval_msec']},{params['solution_voltage']},"
                                f"{console_temp_C},{tx_temp_C},{ambient_temp_C},"
                                f"{hv_set_V},{hv_out_V},{status_message}\n")
                    csv_file.write(csv_line)
                except Exception as e:
                    logger.exception(f"Failed to write to CSV log: {e}")
                    # Consider closing file or stopping CSV logging if errors persist

            if shutdown_event.is_set():
                logger.info("Shutdown event detected by logging thread. Stopping sonication.")
                try:
                    if interface.txdevice.is_trigger_active(): # Check if trigger is active before stopping
                         interface.txdevice.stop_trigger()
                         logger.info("TX trigger stopped by logging thread due to safety event.")
                except Exception as e:
                    logger.exception(f"Error stopping TX trigger from logging thread: {e}")
                break # Exit loop

            # Wait for the next log interval
            # This sleep should be adjusted for the time taken by the loop itself
            loop_end_ns = time.monotonic_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_duration_s = max(0, params['log_interval_sec'] - loop_duration_s)
            if stop_event.wait(timeout=sleep_duration_s): # Wait with timeout, checking stop_event
                 logger.info("Stop event received during sleep, exiting logging loop.")
                 break
            loop_count += 1
        logger.info(f"Temperature logging thread '{thread_name}' finished.")

# ## 7. Hardware Setup, Connection, and Solution Programming
# This cell initializes the LIFUInterface, connects to hardware, sets up power (if not external), and programs the solution.

can_run_experiment = False
if transducer and solution and lifu_db: # Ensure previous cell ran successfully
    logger.info("--- Initiating Hardware Setup ---")

    # Store config in a dict to pass to thread
    thread_params = {'console_shutoff_temp_C': console_shutoff_temp_C,
                    'tx_shutoff_temp_C': tx_shutoff_temp_C,
                    'ambient_shutoff_temp_C': ambient_shutoff_temp_C,
                    'rapid_temp_shutoff_C': rapid_temp_shutoff_C,
                    'rapid_temp_shutoff_seconds': rapid_temp_shutoff_seconds,
                    'rapid_temp_increase_per_second_shutoff_C': rapid_temp_increase_per_second_shutoff_C}
    thread_params['frequency_kHz'] = frequency_kHz
    thread_params['pulse_duration_msec'] = pulse_duration_msec
    thread_params['pulse_interval_msec'] = pulse_interval_msec
    thread_params['solution_voltage'] = solution_voltage
    thread_params['log_interval_sec'] = log_interval_sec
    thread_params['use_external_power_supply'] = use_external_power_supply

    try:
        lifu_interface_global = LIFUInterface()
        logger.info("LIFUInterface initialized.")

        tx_connected, hv_connected = lifu_interface_global.is_device_connected()
        logger.info(f"Initial connection status: TX={tx_connected}, HV={hv_connected}")

        if not use_external_power_supply and not tx_connected and hv_connected:
            logger.warning("TX device not connected. Attempting to turn on 12V via console...")
            lifu_interface_global.hvcontroller.turn_12v_on()
            time.sleep(3) # Allow time for TX to power up and enumerate
            logger.info("Re-initializing LIFU interface after 12V power up...")
            lifu_interface_global.stop_monitoring() # Stop old instance if any async parts were active
            del lifu_interface_global
            lifu_interface_global = LIFUInterface()
            tx_connected, hv_connected = lifu_interface_global.is_device_connected()
            logger.info(f"New connection status: TX={tx_connected}, HV={hv_connected}")

        if not tx_connected:
            raise ConnectionError("TX device is NOT connected. Cannot proceed.")
        if not use_external_power_supply and not hv_connected:
            raise ConnectionError("HV Controller (Console) is NOT connected, but internal power supply is selected. Cannot proceed.")

        logger.info("Performing basic hardware checks (ping, version)...")
        if not lifu_interface_global.txdevice.ping():
            raise RuntimeError("TX device ping failed.")
        logger.info(f"TX Firmware: {lifu_interface_global.txdevice.get_version()}")
        if not use_external_power_supply:
            if not lifu_interface_global.hvcontroller.ping():
                raise RuntimeError("HV controller ping failed.")
            logger.info(f"HV Firmware: {lifu_interface_global.hvcontroller.get_version()}")

        num_tx_chips = lifu_interface_global.txdevice.enum_tx7332_devices()
        expected_chips = num_modules * 2 # Assuming each module has 2 TX7332 chips (e.g. for 32els/chip)
                                         # This logic might need adjustment based on hardware specifics
        # This check might be too strict if num_modules definition is loose
        # if num_tx_chips != expected_chips:
        #    logger.warning(f"Enumerated {num_tx_chips} TX7332 chips, expected {expected_chips} for {num_modules} modules. Check config.")
        # else:
        #    logger.info(f"Correctly enumerated {num_tx_chips} TX7332 chips.")


        # Program the solution
        profile_index = 1
        trigger_mode_hw = "continuous" # For watertank, usually continuous

        logger.info(f"Programming solution to profile {profile_index} with trigger mode '{trigger_mode_hw}'...")
        if use_external_power_supply:
            # lifu_interface_global.check_solution(solution) # This method might not exist or be needed
            sol_dict = solution.to_dict()
            lifu_interface_global.txdevice.set_solution(
                pulse=sol_dict['pulse'], delays=sol_dict['delays'],
                apodizations=sol_dict['apodizations'], sequence=sol_dict['sequence'],
                voltage=sol_dict['voltage'], trigger_mode=trigger_mode_hw,
                profile_index=profile_index, profile_increment=False
            )
            logger.info("Solution programmed to TX device. External power supply selected.")
            logger.warning("Ensure external HV power supply is ON and set to an appropriate voltage for the solution_voltage scaling factor.")
        else:
            # Internal power supply: set_solution can also manage HV
            # For safety, explicitly set HV to the solution_voltage (which is a bit confusingly named here)
            # The 'solution.voltage' is more of a relative amplitude for the TX.
            # The actual HV console voltage should be set based on calibration/desired output.
            # For this example, let's assume solution_voltage is the target HV console rail voltage.
            # THIS IS A SIMPLIFICATION AND MAY NEED ADJUSTMENT FOR REAL EXPERIMENTS.
            target_hv_console_voltage = solution_voltage
            logger.info(f"Setting HV Console voltage to: {target_hv_console_voltage}V")
            if not lifu_interface_global.hvcontroller.set_voltage(target_hv_console_voltage):
                 raise RuntimeError(f"Failed to set HV console voltage to {target_hv_console_voltage}V.")

            lifu_interface_global.set_solution(
                solution=solution, profile_index=profile_index,
                profile_increment=False, trigger_mode=trigger_mode_hw,
                turn_hv_on=True # Let set_solution turn on HV after programming
            )
            logger.info("Solution programmed and HV power turned ON via console.")
            time.sleep(1) # Allow HV to stabilize
            logger.info(f"  HV Status: {'ON' if lifu_interface_global.hvcontroller.get_hv_status() else 'OFF'}")
            logger.info(f"  HV Setpoint: {lifu_interface_global.hvcontroller.get_voltage()}V")
            logger.info(f"  HV Measured Output: {lifu_interface_global.hvcontroller.get_voltage_out()}V")

        logger.info("Hardware setup and solution programming complete.")
        can_run_experiment = True

    except Exception as e:
        logger.exception(f"Error during hardware setup: {e}")
        if lifu_interface_global and not use_external_power_supply and lifu_interface_global.hvcontroller.is_connected():
            logger.info("Attempting to turn off HV and 12V due to setup error...")
            lifu_interface_global.hvcontroller.turn_hv_off()
            lifu_interface_global.hvcontroller.turn_12v_off()
        lifu_interface_global = None # Invalidate on error
        can_run_experiment = False
else:
    logger.exception("Prerequisites for hardware setup (transducer, solution, DB) not met. Please run previous cells successfully.")
    can_run_experiment = False

# ## 8. Experiment Operation Control
#
# **Read Carefully Before Running The Next Cell ("Start Experiment"):**
# *   This cell will start the ultrasound emission if hardware setup was successful.
# *   Temperature monitoring will begin in a separate thread.
# *   The experiment will run until:
#     1.  The "Experiment Timeout" (configured above) is reached.
#     2.  A temperature safety limit is breached, triggering an automatic shutdown.
#     3.  You manually interrupt the Jupyter kernel (e.g., Kernel -> Interrupt).
#     4.  You run the "Manual Stop" cell provided further below (if using ipywidgets or similar).
# *   **Monitor the log output closely.**

# ### 8.1. Start Experiment Cell
# **⚠️ RUNNING THIS CELL WILL START ULTRASOUND EMISSION IF SETUP WAS SUCCESSFUL. ⚠️**

# +
experiment_active = False
experiment_start_time = 0

if can_run_experiment and lifu_interface_global:
    logger.info("--- Starting Watertank Experiment ---")
    operation_shutdown_event_global.clear() # Ensure it's clear before start
    stop_logging_event_global.clear()

    # Prepare CSV filename
    csv_log_file = None
    if log_to_csv_file:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        csv_log_file = log_output_path / f"{timestamp_str}_{frequency_kHz:.0f}kHz_{solution_voltage:.0f}V_TempLog.csv"

    # Start temperature logging thread
    temperature_thread_global = threading.Thread(
        target=temperature_logging_thread_func,
        args=(
            lifu_interface_global,
            stop_logging_event_global,
            operation_shutdown_event_global,
            csv_log_file,
            thread_params # Pass the configuration dict
        ),
        name="TempLoggerThread",
        daemon=True # Allow main program to exit even if thread is running (though we join it)
    )
    temperature_thread_global.start()
    logger.info("Temperature logging thread started.")

    # Start ultrasound emission
    try:
        if lifu_interface_global.txdevice.start_trigger():
            logger.info("✅ Ultrasound trigger STARTED successfully.")
            experiment_active = True
            experiment_start_time = time.monotonic()

            # Display information to user
            # display(Markdown(f"**Experiment active! Monitoring temperatures.**"))
            # display(Markdown(f"Timeout in: {experiment_timeout_minutes} minutes. Logging to console and CSV (if enabled)."))
            # display(Markdown(f"To stop manually: Interrupt Kernel OR run 'Manual Stop' cell if available."))
            print("Experiment active! Monitoring temperatures.")
            print(f"Timeout in: {experiment_timeout_minutes} minutes. Logging to console and CSV (if enabled).")
            print("To stop manually: Interrupt Kernel.")


            # Main experiment loop: waits for shutdown event or timeout
            while experiment_active and not operation_shutdown_event_global.is_set():
                if (time.monotonic() - experiment_start_time) > (experiment_timeout_minutes * 60):
                    logger.warning("Experiment timeout reached. Initiating shutdown.")
                    operation_shutdown_event_global.set() # Signal shutdown
                    break
                # Sleep briefly, allowing other threads to run and check shutdown event
                # This makes the loop responsive to the shutdown event.
                if operation_shutdown_event_global.wait(timeout=0.5): # Check event every 0.5s
                    logger.info("Operation shutdown event received in main loop.")
                    break

            if operation_shutdown_event_global.is_set():
                 logger.info("Shutdown event was set (temperature limit, error, timeout, or manual stop).")

        else:
            logger.exception("❌ Failed to start ultrasound trigger.")
            operation_shutdown_event_global.set() # Signal threads to stop

    except Exception as e:
        logger.exception(f"Exception during experiment operation: {e}")
        operation_shutdown_event_global.set() # Signal threads to stop
    finally:
        experiment_active = False
        logger.info("--- Experiment Main Loop Ended ---")
        # Cleanup will be handled in the next dedicated cell.
else:
    logger.exception("Cannot start experiment: Hardware setup not complete or interface not available.")
# -

# ### 8.2. Manual Stop (Informational)
#
# If you need to stop the experiment *before* a timeout or safety shutdown:
# 1.  **Interrupt the Jupyter Kernel:** Go to "Kernel" in the menu and select "Interrupt". This is the most reliable way to stop execution of the above cell.
# 2.  **If `ipywidgets` were used for a button:** A dedicated "Stop" button cell would set `operation_shutdown_event_global.set()`. This is not implemented here to keep dependencies minimal but is a common pattern for interactive notebooks.
#
# After interrupting or if the experiment completes/shuts down, proceed to the **Cleanup** cell below.

# ## 9. Cleanup
# **Run this cell after the experiment has finished or been stopped** to ensure hardware is safely shut down and resources are released.

# +
logger.info("--- Initiating Experiment Cleanup ---")

# Signal logging thread to stop, regardless of how experiment ended
stop_logging_event_global.set()
if not operation_shutdown_event_global.is_set(): # If not already set by safety/timeout
    operation_shutdown_event_global.set() # Ensure it's set to stop any lingering main loop checks

if lifu_interface_global:
    try:
        # Stop TX trigger if it might still be active
        # (Temperature thread might have already stopped it on safety event)
        if lifu_interface_global.txdevice.is_connected():
            logger.info("Ensuring TX trigger is stopped...")
            if lifu_interface_global.txdevice.is_trigger_active(): # Requires is_trigger_active method
                if lifu_interface_global.txdevice.stop_trigger():
                    logger.info("TX trigger stopped successfully via cleanup.")
                else:
                    logger.warning("Failed to stop TX trigger during cleanup. Power cycle may be needed if issues persist.")
            else:
                logger.info("TX trigger was already inactive.")
        else:
            logger.warning("TX device not connected during cleanup, cannot stop trigger.")

    except AttributeError:
        logger.warning("TXDevice does not have 'is_trigger_active' method. Attempting stop_trigger regardless.")
        try:
            if lifu_interface_global.txdevice.is_connected():
                lifu_interface_global.txdevice.stop_trigger()
                logger.info("Attempted to stop TX trigger (is_trigger_active not available).")
        except Exception as e_stop:
            logger.exception(f"Error trying to stop trigger during cleanup: {e_stop}")
    except Exception as e:
        logger.exception(f"Error during TX trigger stop in cleanup: {e}")

    # Wait for temperature logging thread to finish
    if temperature_thread_global and temperature_thread_global.is_alive():
        logger.info("Waiting for temperature logging thread to complete...")
        temperature_thread_global.join(timeout=params.get('log_interval_sec', 1.0) * 3) # Wait up to 3 log intervals
        if temperature_thread_global.is_alive():
            logger.warning("Temperature logging thread did not exit cleanly after join timeout.")
        else:
            logger.info("Temperature logging thread finished.")

    # Turn off console HV and 12V if not using external power supply
    if not use_external_power_supply and lifu_interface_global.hvcontroller.is_connected():
        logger.info("Turning off HV and 12V power via console...")
        try:
            if lifu_interface_global.hvcontroller.get_hv_status():
                lifu_interface_global.hvcontroller.turn_hv_off()
                logger.info("HV power turned OFF.")
            if lifu_interface_global.hvcontroller.get_12v_status():
                lifu_interface_global.hvcontroller.turn_12v_off()
                logger.info("12V power turned OFF.")
        except Exception as e:
            logger.exception(f"Error turning off console power: {e}")

    # Clean up interface
    logger.info("Stopping LIFUInterface monitoring and deleting instance...")
    lifu_interface_global.stop_monitoring() # Important if any async parts were active
    del lifu_interface_global
    lifu_interface_global = None
    logger.info("LIFUInterface instance deleted.")

else:
    logger.info("LIFUInterface not available for cleanup (was not initialized or already cleaned up).")

logger.info("--- Cleanup Complete ---")
# -

# ## 10. Conclusion
#
# This notebook provided a detailed workflow for continuous watertank operations, including:
# *   Parameter configuration.
# *   Solution definition and hardware programming.
# *   Threaded temperature monitoring with CSV logging.
# *   Automated safety shutdowns for temperature and experiment duration.
#
# This forms a robust basis for many automated ultrasound experiments. Remember to always adapt parameters and safety checks to your specific setup and requirements.
#
# **Further Exploration:**
# *   Analyze the generated CSV temperature logs.
# *   Integrate more sophisticated `Solution` generation from Notebook 03.
# *   Explore advanced triggering or synchronization if needed for your experiments.

# End of Notebook 09
