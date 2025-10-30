from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime

if os.name == 'nt':
    import msvcrt
else:
    import select

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

##########################################################################
#                       CONFIGURATION SETTINGS                           #
##########################################################################

voltage_start = 5 # Starting voltage value PER RAIL (not peak to peak)
voltage_step = 5 # Voltage increase for each step from start to end by this amount

frequency_kHz = 400 # Frequency in kHz (150 or 400)
number_of_boards = 2 # Number of boards in the system (1 or 2)

##########################################################################
#                                                                        #
##########################################################################



profile_choices = {
    "low_duty_cycle": {
        "pulse_duration_msec": 5,
        "pulse_interval_msec": 100,
        "voltage_end": 65
    },
    "medium_duty_cycle": {
        "pulse_duration_msec": 25,
        "pulse_interval_msec": 100,
        "voltage_end": 45
    },
    "high_duty_cycle": {
        "pulse_duration_msec": 50,
        "pulse_interval_msec": 100,
        "voltage_end": 30
    }
}

try:
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prevent duplicate handlers and cluttered terminal output
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        file_handle = logging.FileHandler(f"test_console_voltage_{timestamp}.log", mode='w')
        logger.addHandler(file_handle)

    # Turn dict into an indexed list of (name, config)
    profiles_list = list(profile_choices.items())
    max_name_len = max(len(name) for name in profile_choices) # to print neatly

    # Display profiles list
    logger.info("Available Test Profiles to Select:")
    for i, (name, params) in enumerate(profiles_list, start=1):
        logger.info(
            f"{i}. {name:<{max_name_len}} | "
            f"Pulse Duration: {params['pulse_duration_msec']:>3} ms, "
            f"Pulse Interval: {params['pulse_interval_msec']:>3} ms, "
            f"Voltage End: {params['voltage_end']:>3} V"
        )

    # Wait for user to select profile
    while True:
        choice = input("Select a Test Profile by number (1, 2, 3): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(profiles_list):
                profile_name, current = profiles_list[idx]

                logger.info(f"Selected Profile: {profile_name}")
                logger.info(f"Pulse Duration: {current['pulse_duration_msec']} ms")
                logger.info(f"Pulse Interval: {current['pulse_interval_msec']} ms")
                logger.info(f"Voltage End: {current['voltage_end']} V")
                break
            else:
                print("Invalid number! Try again...")
        else:
            print("Please enter a number only! Try again...")

    voltage_end = current['voltage_end']
    pulse_duration_msec = current['pulse_duration_msec']
    pulse_interval_msec = current['pulse_interval_msec']

    log_interval = 1  # seconds; you can adjust this variable as needed

    console_shutoff_temp_C = 70.0 # Console shutoff temperature in Celsius
    tx_shutoff_temp_C = 70.0 # TX device shutoff temperature in Celsius
    ambient_shutoff_temp_C = 70.0 # Ambient shutoff temperature in Celsius

    # Fail-safe parameters if the temperature jumps too fast
    rapid_temp_shutoff_C = 40 # Cutoff temperature in Celsius if it jumps too fast
    rapid_temp_shutoff_seconds = 5 # Time in seconds to reach rapid temperature shutoff
    rapid_transmitter_temp_increase_per_second_shutoff_C = 3 # Rapid temperature climbing shutoff in Celsius
    rapid_console_temp_increase_per_second_shutoff_C = 5 # Rapid temperature climbing shutoff in Celsius

    # Create delays array with proper shape for 2D indexing
    delays = np.zeros((1, 64*number_of_boards))

    apodizations = np.ones((1, 64*number_of_boards))

    logger.info("Starting LIFU Test Console Voltage Script...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()

    if not tx_connected:
        logger.warning("TX device not connected. Attempting to turn on 12V...")
        interface.hvcontroller.turn_12v_on()

        # Give time for the TX device to power up and enumerate over USB
        time.sleep(2)

        # Cleanup and recreate interface to reinitialize USB devices
        interface.stop_monitoring()
        del interface
        time.sleep(1)  # Short delay before recreating

        logger.info("Reinitializing LIFU interface after powering 12V...")
        interface = LIFUInterface()

        # Re-check connection
        tx_connected, hv_connected = interface.is_device_connected()

    if hv_connected:
        logger.info(f"  HV Connected: {hv_connected}")
    else:
        logger.error("!! HV NOT fully connected. !!")
        sys.exit(1)

    if tx_connected:
        logger.info(f"  TX Connected: {tx_connected}")
        logger.info(">>> LIFU Device fully connected. <<<")
    else:
        logger.error("!! TX NOT fully connected. !!")
        sys.exit(1)

    stop_logging = False  # flag to signal the logging thread to stop

    def monitor_temperature():
        start = time.time()
        shutdown = False

        prev_tx_temp = None
        prev_amb_temp = None
        prev_con_temp = None

        while not (stop_logging or shutdown):
            # Timer for initial rapid temperature increase
            within_initial_time_threshold = (time.time() - start) < rapid_temp_shutoff_seconds

            ## Check for too high of a temperature increase between readings
            # Console general temperature increase (bypass if using external power supply)
            if prev_con_temp is None:
                prev_con_temp = interface.hvcontroller.get_temperature1()
                prev_con_temp = 0
            # con_temp = interface.hvcontroller.get_temperature1()
            con_temp = 0
            if (con_temp - prev_con_temp) > rapid_console_temp_increase_per_second_shutoff_C:
                logger.warning(f"Console temperature rose from {prev_con_temp}°C to {con_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                shutdown=True
            else:
                prev_con_temp = con_temp

            # TX device general temperature increase
            if prev_tx_temp is None:
                prev_tx_temp = interface.txdevice.get_temperature()
            tx_temp = interface.txdevice.get_temperature()
            if (tx_temp - prev_tx_temp) > rapid_transmitter_temp_increase_per_second_shutoff_C:
                logger.warning(f"TX device temperature rose from {prev_tx_temp}°C to {tx_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                shutdown=True
            else:
                prev_tx_temp = tx_temp

            # Ambient temperature general increase
            if prev_amb_temp is None:
                prev_amb_temp = interface.txdevice.get_ambient_temperature()
            amb_temp = interface.txdevice.get_ambient_temperature()
            if (amb_temp - prev_amb_temp) > rapid_transmitter_temp_increase_per_second_shutoff_C:
                logger.warning(f"Ambient temperature rose from {prev_amb_temp}°C to {amb_temp}°C (above {rapid_temp_increase_per_second_shutoff_C}°C threshold) within {log_interval}s.")
                shutdown=True
            else:
                prev_amb_temp = amb_temp

            # Check for initial rapid temperature increase
            if (within_initial_time_threshold):
                if (con_temp > rapid_temp_shutoff_C):
                    logger.warning(f"Console temperature {con_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    shutdown=True
                elif (tx_temp > rapid_temp_shutoff_C):
                    logger.warning(f"TX device temperature {tx_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    shutdown=True
                elif (amb_temp > rapid_temp_shutoff_C):
                    logger.warning(f"Ambient temperature {amb_temp}°C exceeds rapid shutoff threshold of {rapid_temp_shutoff_C}°C within {rapid_temp_shutoff_seconds}s.")
                    shutdown=True

            # Check if any temperature exceeds the shutoff threshold
            if (con_temp > console_shutoff_temp_C):
                logger.warning(f"Console temperature {con_temp}°C exceeds shutoff threshold {console_shutoff_temp_C}°C.")
                shutdown=True
            if tx_temp > tx_shutoff_temp_C:
                logger.warning(f"TX device temperature {tx_temp}°C exceeds shutoff threshold {tx_shutoff_temp_C}°C.")
                shutdown=True
            if amb_temp > ambient_shutoff_temp_C:
                logger.warning(f"Ambient temperature {amb_temp}°C exceeds shutoff threshold {ambient_shutoff_temp_C}°C.")
                shutdown=True

            if shutdown:
                logger.warning("Manually turning off High Voltage and 12V due to temperature safety limits...")
                interface.hvcontroller.turn_hv_off()
                interface.hvcontroller.turn_12v_off()
                break
            time.sleep(log_interval)
        sys.exit(0)

    # Verify communication with the devices
    if not interface.txdevice.ping():
        logger.error("Failed to ping the transmitter device.")
        sys.exit(1)

    if not interface.hvcontroller.ping():
        logger.error("Failed to ping the console device.")
        sys.exit(1)

    try:
        console_firmware_version = interface.hvcontroller.get_version()
        # logger.info(f"Console Firmware Version: {console_firmware_version}")
    except Exception as e:
        logger.error(f"Error querying console firmware version: {e}")
    try:
        tx_firmware_version = interface.txdevice.get_version()
        # logger.info(f"TX Firmware Version: {tx_firmware_version}")
    except Exception as e:
        logger.error(f"Error querying TX firmware version: {e}")

    logger.info("Enumerate TX7332 chips")
    num_tx_devices = interface.txdevice.enum_tx7332_devices()
    if num_tx_devices == 0:
        raise ValueError("No TX7332 devices found.")
    elif num_tx_devices == number_of_boards*2:
        logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
        numelements = 32*num_tx_devices
    else:
        raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{number_of_boards}")

    pulse = Pulse(frequency=frequency_kHz*1e3, duration=pulse_duration_msec*1e-3)

    sequence = Sequence(
        pulse_interval=pulse_interval_msec*1e-3,
        pulse_count=int(60/(pulse_interval_msec*1e-3)),
        pulse_train_interval=0,
        pulse_train_count=1
    )

    # Create pin order for dummy array (sequential ordering)
    pin_order = np.arange(0, 64*number_of_boards)

    solution = Solution(
        delays = delays[:, pin_order],
        apodizations = apodizations[:, pin_order],
        pulse = pulse,
        voltage=voltage_start,
        sequence = sequence
    )

    duty_cycle = round(((pulse_duration_msec/pulse_interval_msec) * 100), 2)

    trigger_mode = "continuous"

    profile_index = 1

    interface.set_solution(
        solution=solution,
        profile_index=profile_index,
        profile_increment=True,
        trigger_mode=trigger_mode)

    duty_cycle = round((pulse_duration_msec / pulse_interval_msec) * 100, 2)
    if duty_cycle > 50:
        logger.warning("❗❗ Duty cycle is above 50% ❗❗")

    logger.info(
        f"""\n
    ================================================ User Settings =================================================
    {'Voltage Per Rail Starting Value:':33} {voltage_start:>3}V (Peak to Peak: {voltage_start*2}V)
    {'Voltage Per Rail Ending Value:':33} {voltage_end:>3}V{'❗❗ WARNING: Exceeds safe limit of 65V! ❗❗' if voltage_end > 65 else ''} (Peak to Peak: {voltage_end*2}V)
    Voltage Step Size: {voltage_step}V

    Number of Boards Connected: {number_of_boards}
    Frequency: {frequency_kHz} kHz
    Pulse Duration: {pulse_duration_msec} ms
    Pulse Interval: {pulse_interval_msec} ms
    Duty Cycle: {duty_cycle} %
    ================================================================================================================\n
    """
    )

    def turn_off_console_and_tx():
        logger.info("Attempting to turn off High Voltage...")
        interface.hvcontroller.turn_hv_off()
        if interface.hvcontroller.get_hv_status():
            logger.error("High Voltage is still on.")
        else:
            logger.info("High Voltage successfully turned off.")

        logger.info("Attempting to turn off 12V...")
        interface.hvcontroller.turn_12v_off()
        if interface.hvcontroller.get_12v_status():
            logger.error("12V is still on.")
        else:
            logger.info("12V successfully turned off.")

    shutdown_event = threading.Event()

    def input_wrapper():
        if os.name == 'nt':
            while not shutdown_event.is_set():
                if msvcrt.kbhit():  # Check if a key has been pressed
                    char = msvcrt.getch()  # Read the key
                    if char == b'\r':  # Check if the Enter key was pressed
                        break
                time.sleep(0.1)  # Small delay to prevent high CPU usage
        else:
            while not shutdown_event.is_set():
                # Use select to check if input is available
                ready, _, _ = select.select([], [], [], 0.1)  # Timeout of 0.1 seconds
                if ready:
                    sys.stdin.readline() # Read the input
                    break

    t = threading.Thread(target=monitor_temperature)
    # user_input = threading.Thread(target=input_wrapper)

    input("Press enter to start voltage test:")
    logger.info("Starting Voltage Test...")
    logger.info(f"Setting Voltage to {voltage_start}V")
    if interface.start_sonication():
        t.start()

        logger.info(f"Voltage set to {voltage_start}V")
        for voltage in range(voltage_start, voltage_end+voltage_step, voltage_step):
            if voltage+voltage_step <= voltage_end:
                user_input = input(f"Press enter to continue to next voltage step of {voltage+voltage_step}V "
                      "(or press 'q' to quit): ")
                if user_input.lower() == 'q':
                    logger.info("User selected 'q', exiting voltage test...")
                    break
            else:
                input("Press enter to exit:")
                break

            interface.hvcontroller.turn_hv_off()
            time.sleep(2)  # Wait for HV to turn off
            logger.info(f"Setting Voltage to {voltage+voltage_step}V")
            if not interface.hvcontroller.set_voltage(voltage+voltage_step):
                logger.error(f"Failed to set voltage to {voltage+voltage_step}V")
                break
            interface.hvcontroller.turn_hv_on()
            time.sleep(5)
            logger.info(f"Voltage set to {voltage+voltage_step}V")

        shutdown_event.set()  # Signal the logging thread to stop
        stop_logging = True

        t.join()  # Wait for the logging thread to finish

        time.sleep(0.5)  # Give the logging thread time to finish
        if interface.stop_sonication():
            logger.info("Trigger stopped successfully.")
        else:
            logger.error("Failed to stop trigger.")
    else:
        logger.error("Failed to get trigger setting.")

    turn_off_console_and_tx()
except KeyboardInterrupt:
    logger.info("Keyboard interrupt received. Exiting...")
    stop_logging = True
    t.join()
    turn_off_console_and_tx()
    shutdown_event.set()
