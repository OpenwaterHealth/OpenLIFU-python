#!/usr/bin/env python3
"""
LIFU Thermal Stress Test Script

A professional command-line tool for automated thermal stress testing of LIFU devices.
This script connects to the device, configures test parameters, monitors temperatures,
and logs data with automatic safety shutoffs.

Author: OpenLIFU Team
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import select

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

__version__ = "1.0.0"


class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode characters safely on Windows."""

    def format(self, record):
        """Format the log record, removing unsupported Unicode characters."""
        try:
            msg = super().format(record)
            # Remove or replace emojis/non-ASCII characters for Windows compatibility
            return msg.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return super().format(record)


class ThermalStressTest:
    """Main class for managing LIFU thermal stress testing."""

    def __init__(self, args):
        """Initialize the thermal stress test with command-line arguments."""
        self.args = args
        self.interface = None
        self.stop_logging = False
        self.shutdown_event = threading.Event()
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Configure logging with appropriate format and level."""
        logger = logging.getLogger(__name__)

        # Set log level based on verbosity
        if self.args.verbose:
            logger.setLevel(logging.DEBUG)
        elif self.args.quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create safe formatter for Windows compatibility
        formatter = SafeFormatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler - create run log
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_filename = output_dir / f"{self.run_timestamp}_run_log.txt"
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Run log will be saved to: {log_filename}")

        return logger

    def connect_device(self):
        """Connect to the LIFU device and verify connection."""
        self.logger.info("Starting LIFU Test Script...")
        self.interface = LIFUInterface()
        tx_connected, hv_connected = self.interface.is_device_connected()

        if not self.args.external_power and not tx_connected:
            self.logger.warning("TX device not connected. Attempting to turn on 12V...")
            self.interface.hvcontroller.turn_12v_on()
            time.sleep(2)

            # Cleanup and recreate interface
            try:
                self.interface.stop_monitoring()
                del self.interface
            except Exception as e:
                self.logger.warning(f"Error during interface cleanup: {e}")

            time.sleep(1)

            self.logger.info("Reinitializing LIFU interface after powering 12V...")
            self.interface = LIFUInterface()
            tx_connected, hv_connected = self.interface.is_device_connected()

        # Verify connections
        if not self.args.external_power:
            if hv_connected:
                self.logger.info(f"  HV Connected: {hv_connected}")
            else:
                self.logger.error("HV NOT fully connected.")
                sys.exit(1)
        else:
            self.logger.info("  Using external power supply")

        if tx_connected:
            self.logger.info(f"  TX Connected: {tx_connected}")
            self.logger.info("LIFU Device fully connected.")
        else:
            self.logger.error("TX NOT fully connected.")
            sys.exit(1)

    def verify_communication(self):
        """Verify communication with devices and display firmware versions."""
        if not self.interface.txdevice.ping():
            self.logger.error("Failed to ping the transmitter device.")
            sys.exit(1)

        if not self.args.external_power and not self.interface.hvcontroller.ping():
            self.logger.error("Failed to ping the console device.")
            sys.exit(1)

        # Get firmware versions
        if not self.args.external_power:
            try:
                console_firmware_version = self.interface.hvcontroller.get_version()
                self.logger.info(f"Console Firmware Version: {console_firmware_version}")
            except Exception as e:
                self.logger.error(f"Error querying console firmware version: {e}")

        try:
            tx_firmware_version = self.interface.txdevice.get_version()
            self.logger.info(f"TX Firmware Version: {tx_firmware_version}")
        except Exception as e:
            self.logger.error(f"Error querying TX firmware version: {e}")

    def enumerate_devices(self):
        """Enumerate TX7332 devices and verify count."""
        self.logger.info("Enumerate TX7332 chips")
        num_tx_devices = self.interface.txdevice.enum_tx7332_devices()

        if num_tx_devices == 0:
            raise ValueError("No TX7332 devices found.")
        elif num_tx_devices == self.args.num_modules * 2:
            self.logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
            return 32 * num_tx_devices
        else:
            raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.args.num_modules}")

    def configure_solution(self, numelements):
        """Configure the pulse solution with specified parameters."""
        if not self.args.external_power:
            if self.interface.hvcontroller.set_voltage(self.args.voltage):
                self.logger.info("High Voltage set successfully.")
            else:
                self.logger.error("Failed to set High Voltage.")
                sys.exit(1)

        pulse = Pulse(
            frequency=self.args.frequency * 1e3,
            duration=self.args.duration * 1e-3
        )

        delays = np.zeros(numelements)
        apodizations = np.ones(numelements)

        sequence = Sequence(
            pulse_interval=self.args.interval * 1e-3,
            pulse_count=int(60 / (self.args.interval * 1e-3)),
            pulse_train_interval=0,
            pulse_train_count=self.args.test_duration
        )

        solution = Solution(
            delays=delays,
            apodizations=apodizations,
            pulse=pulse,
            voltage=self.args.voltage,
            sequence=sequence
        )

        sol_dict = solution.to_dict()
        self.interface.txdevice.set_solution(
            pulse=sol_dict['pulse'],
            delays=sol_dict['delays'],
            apodizations=sol_dict['apodizations'],
            sequence=sol_dict['sequence'],
            trigger_mode="continuous",
            profile_index=1,
            profile_increment=True
        )

        # Calculate and warn about duty cycle
        duty_cycle = int((self.args.duration / self.args.interval) * 100)
        if duty_cycle > 50:
            self.logger.warning("WARNING: Duty cycle is above 50%")

        # Display configuration
        peak_to_peak_voltage = self.args.voltage * 2
        self.logger.info(f"""User parameters set:
    Frequency: {self.args.frequency}kHz
    Voltage Per Rail: {self.args.voltage}V
    Voltage Peak to Peak: {peak_to_peak_voltage}V
    Duration: {self.args.duration}ms
    Interval: {self.args.interval}ms
    Duty Cycle: {duty_cycle}%
    Use External Power Supply: {self.args.external_power}
    Initial Temp Safety Shutoff: Increase to {self.args.rapid_temp_shutoff}C within {self.args.rapid_temp_time}s of starting.
    TX Temp Safety Shutoff: Increase of {self.args.rapid_tx_increase}C within {self.args.log_interval}s at any point.
    Console Temp Safety Shutoff: Increase of {self.args.rapid_console_increase}C within {self.args.log_interval}s at any point.""")

    def log_temperature(self):
        """Log temperature readings to CSV file with safety monitoring."""
        start = time.time()

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"{self.run_timestamp}_{self.args.frequency}kHz_{self.args.voltage}V_{self.args.duration}ms_Duration_{self.args.interval}ms_Interval_Temperature_Readings.csv"

        shutdown = False
        prev_tx_temp = None
        prev_amb_temp = None
        prev_con_temp = None

        con_temp = "N/A" if self.args.external_power else None

        try:
            with open(filename, "w") as logfile:
                # Write CSV header
                log_line = "Current Time,Frequency (kHz),Duration (ms),Interval (ms),Voltage (Per Rail),Voltage (Peak to Peak),Console Temperature (C),Transmitter Temperature (C),Ambient Temperature (C)\n"
                logfile.write(log_line)
                logfile.flush()

                while not (self.stop_logging or shutdown or self.shutdown_event.is_set()):
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    within_initial_time_threshold = (time.time() - start) < self.args.rapid_temp_time

                    # Check console temperature
                    if not self.args.external_power:
                        try:
                            if prev_con_temp is None:
                                prev_con_temp = self.interface.hvcontroller.get_temperature1()
                            con_temp = self.interface.hvcontroller.get_temperature1()
                            if (con_temp - prev_con_temp) > self.args.rapid_console_increase:
                                self.logger.warning(f"Console temperature rose from {prev_con_temp}C to {con_temp}C (above {self.args.rapid_console_increase}C threshold) within {self.args.log_interval}s.")
                                log_line = f"{current_time},SHUTDOWN,Console temperature exceeded rapid temp increase shutoff threshold\n"
                                shutdown = True
                            else:
                                prev_con_temp = con_temp
                        except Exception as e:
                            self.logger.error(f"Error reading console temperature: {e}")
                            break

                    # Check TX temperature
                    try:
                        if prev_tx_temp is None:
                            prev_tx_temp = self.interface.txdevice.get_temperature()
                        tx_temp = self.interface.txdevice.get_temperature()
                        if (tx_temp - prev_tx_temp) > self.args.rapid_tx_increase:
                            self.logger.warning(f"TX device temperature rose from {prev_tx_temp}C to {tx_temp}C (above {self.args.rapid_tx_increase}C threshold) within {self.args.log_interval}s.")
                            log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded rapid temp increase shutoff threshold\n"
                            shutdown = True
                        else:
                            prev_tx_temp = tx_temp
                    except Exception as e:
                        self.logger.error(f"Error reading TX temperature: {e}")
                        break

                    # Check ambient temperature
                    try:
                        if prev_amb_temp is None:
                            prev_amb_temp = self.interface.txdevice.get_ambient_temperature()
                        amb_temp = self.interface.txdevice.get_ambient_temperature()
                        if (amb_temp - prev_amb_temp) > self.args.rapid_tx_increase:
                            self.logger.warning(f"Ambient temperature rose from {prev_amb_temp}C to {amb_temp}C (above {self.args.rapid_tx_increase}C threshold) within {self.args.log_interval}s.")
                            log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded rapid temp increase shutoff threshold\n"
                            shutdown = True
                        else:
                            prev_amb_temp = amb_temp
                    except Exception as e:
                        self.logger.error(f"Error reading ambient temperature: {e}")
                        break

                    # Check for initial rapid temperature increase
                    if within_initial_time_threshold:
                        if not self.args.external_power and (con_temp != "N/A" and con_temp > self.args.rapid_temp_shutoff):
                            self.logger.warning(f"Console temperature {con_temp}C exceeds rapid shutoff threshold of {self.args.rapid_temp_shutoff}C within {self.args.rapid_temp_time}s.")
                            log_line = f"{current_time},SHUTDOWN,Console temperature exceeded rapid shutoff threshold\n"
                            shutdown = True
                        elif tx_temp > self.args.rapid_temp_shutoff:
                            self.logger.warning(f"TX device temperature {tx_temp}C exceeds rapid shutoff threshold of {self.args.rapid_temp_shutoff}C within {self.args.rapid_temp_time}s.")
                            log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded shutoff threshold\n"
                            shutdown = True
                        elif amb_temp > self.args.rapid_temp_shutoff:
                            self.logger.warning(f"Ambient temperature {amb_temp}C exceeds rapid shutoff threshold of {self.args.rapid_temp_shutoff}C within {self.args.rapid_temp_time}s.")
                            log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded shutoff threshold\n"
                            shutdown = True

                    # Log current readings
                    peak_to_peak_voltage = self.args.voltage * 2
                    log_line = f"{current_time},{self.args.frequency},{self.args.duration},{self.args.interval},{self.args.voltage},{peak_to_peak_voltage},{con_temp},{tx_temp},{amb_temp}\n"

                    if not self.args.external_power:
                        self.logger.info(f"Console Temp: {con_temp}C TX Temp: {tx_temp}C Ambient Temp: {amb_temp}C")
                    else:
                        self.logger.info(f"Console Temp: {con_temp} TX Temp: {tx_temp}C Ambient Temp: {amb_temp}C")

                    logfile.write(log_line)
                    logfile.flush()

                    # Check absolute temperature thresholds
                    if not self.args.external_power and (con_temp != "N/A" and con_temp > self.args.console_shutoff_temp):
                        self.logger.warning(f"Console temperature {con_temp}C exceeds shutoff threshold {self.args.console_shutoff_temp}C.")
                        log_line = f"{current_time},SHUTDOWN,Console temperature exceeded shutoff threshold\n"
                        shutdown = True
                    if tx_temp > self.args.tx_shutoff_temp:
                        self.logger.warning(f"TX device temperature {tx_temp}C exceeds shutoff threshold {self.args.tx_shutoff_temp}C.")
                        log_line = f"{current_time},SHUTDOWN,TX device temperature exceeded shutoff threshold\n"
                        shutdown = True
                    if amb_temp > self.args.ambient_shutoff_temp:
                        self.logger.warning(f"Ambient temperature {amb_temp}C exceeds shutoff threshold {self.args.ambient_shutoff_temp}C.")
                        log_line = f"{current_time},SHUTDOWN,Ambient temperature exceeded shutoff threshold\n"
                        shutdown = True

                    if shutdown:
                        try:
                            self.interface.txdevice.stop_trigger()
                        except Exception as e:
                            self.logger.error(f"Error stopping trigger: {e}")
                        logfile.write(log_line)
                        logfile.flush()
                        break

                    time.sleep(self.args.log_interval)

            minutes, seconds = divmod(int(time.time() - start), 60)
            self.logger.info(f"Temperature logging stopped after {minutes}:{seconds:02d}.")
            self.logger.info(f"Data saved to \"{filename}\".")

        except Exception as e:
            self.logger.error(f"Error in temperature logging: {e}")
        finally:
            self.shutdown_event.set()

    def input_wrapper(self):
        """Wait for user input to stop the test."""
        try:
            if os.name == 'nt':
                while not self.shutdown_event.is_set():
                    if msvcrt.kbhit():
                        char = msvcrt.getch()
                        if char == b'\r':
                            break
                    time.sleep(0.1)
            else:
                while not self.shutdown_event.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        sys.stdin.readline()
                        break
        except Exception as e:
            self.logger.error(f"Error in input wrapper: {e}")

    def turn_off_console_and_tx(self):
        """Safely turn off console and TX device."""
        if not self.args.external_power:
            try:
                self.logger.info("Attempting to turn off High Voltage...")
                self.interface.hvcontroller.turn_hv_off()
                time.sleep(0.5)
                if self.interface.hvcontroller.get_hv_status():
                    self.logger.error("High Voltage is still on.")
                else:
                    self.logger.info("High Voltage successfully turned off.")
            except Exception as e:
                self.logger.warning(f"Error turning off High Voltage: {e}")

            try:
                self.logger.info("Attempting to turn off 12V...")
                self.interface.hvcontroller.turn_12v_off()
                time.sleep(0.5)
                if self.interface.hvcontroller.get_12v_status():
                    self.logger.error("12V is still on.")
                else:
                    self.logger.info("12V successfully turned off.")
            except Exception as e:
                self.logger.warning(f"Error turning off 12V: {e}")

    def cleanup_interface(self):
        """Safely cleanup the LIFU interface."""
        if self.interface:
            try:
                self.logger.info("Closing device interface...")
                self.interface.stop_monitoring()
                time.sleep(0.5)
                del self.interface
                self.interface = None
            except Exception as e:
                self.logger.warning(f"Issue closing LIFU interface: {e}")

    def run(self):
        """Execute the thermal stress test with graceful shutdown."""
        t_log = None
        t_input = None

        try:
            # Connect and verify
            self.connect_device()
            self.verify_communication()
            numelements = self.enumerate_devices()

            # Configure test
            self.configure_solution(numelements)

            # Wait for user to start
            if not self.args.no_prompt:
                self.logger.info("Press enter to START trigger:")
                input()

            # Enable high voltage
            if not self.args.external_power:
                self.logger.info("Enable High Voltage")
                if not self.interface.hvcontroller.turn_hv_on():
                    self.logger.error("Failed to turn on High Voltage.")
                    return
            else:
                self.logger.info("Using external power supply")

            # Start sonication
            self.logger.info("Starting Trigger...")
            if not self.interface.start_sonication():
                self.logger.error("Failed to start sonication.")
                return

            self.logger.info("Trigger Running...")
            if not self.args.no_prompt:
                self.logger.info("Press enter to STOP trigger:")

            # Start logging and user input threads (daemon threads will auto-terminate)
            t_log = threading.Thread(target=self.log_temperature, daemon=True)
            t_input = threading.Thread(target=self.input_wrapper, daemon=True)

            t_log.start()
            t_input.start()

            # Wait for threads to complete or shutdown signal
            while t_log.is_alive() and t_input.is_alive() and not self.shutdown_event.is_set():
                time.sleep(0.1)

            if not t_input.is_alive():
                self.logger.info("Logging interrupted by user.")
            elif not t_log.is_alive():
                self.logger.info("Logging completed or interrupted by shutdown event.")

        except KeyboardInterrupt:
            self.logger.warning("Keyboard interrupt detected - initiating graceful shutdown...")

        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")

        finally:
            # Signal all threads to stop
            self.shutdown_event.set()
            self.stop_logging = True

            # Wait briefly for threads to finish
            self.logger.info("Waiting for threads to complete...")
            time.sleep(1.0)

            # Stop sonication
            try:
                self.logger.info("Stopping sonication...")
                if self.interface:
                    self.interface.stop_sonication()
            except Exception as e:
                self.logger.warning(f"Error stopping sonication: {e}")

            # Turn off hardware
            try:
                self.logger.info("Turning off console and TX...")
                self.turn_off_console_and_tx()
            except Exception as e:
                self.logger.warning(f"Issue turning off hardware: {e}")

            # Cleanup interface
            self.cleanup_interface()

            self.logger.info("Graceful shutdown complete.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LIFU Thermal Stress Test - Automated testing with temperature monitoring and safety shutoffs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  %(prog)s

  # Run with custom frequency and voltage
  %(prog)s --frequency 500 --voltage 20

  # Run with external power supply
  %(prog)s --external-power

  # Run with custom safety thresholds
  %(prog)s --console-shutoff-temp 80 --tx-shutoff-temp 75

  # Run in quiet mode with custom output directory
  %(prog)s --quiet --output-dir ./test_results
        """
    )

    # Version
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    # Test parameters
    test_group = parser.add_argument_group('Test Parameters')
    test_group.add_argument(
        '-f', '--frequency',
        type=float,
        default=400,
        metavar='KHZ',
        help='Frequency in kHz (default: 400)'
    )
    test_group.add_argument(
        '-v', '--voltage',
        type=float,
        default=15.0,
        metavar='V',
        help='Voltage per rail in Volts (default: 15.0)'
    )
    test_group.add_argument(
        '-d', '--duration',
        type=float,
        default=5,
        metavar='MS',
        help='Pulse duration in milliseconds (default: 5)'
    )
    test_group.add_argument(
        '-i', '--interval',
        type=float,
        default=100,
        metavar='MS',
        help='Pulse repetition interval in milliseconds (default: 100)'
    )
    test_group.add_argument(
        '-m', '--num-modules',
        type=int,
        default=1,
        metavar='N',
        help='Number of modules in the system (default: 1)'
    )
    test_group.add_argument(
        '-t', '--test-duration',
        type=int,
        default=60,
        metavar='MIN',
        help='Test duration in minutes (default: 60)'
    )

    # Power configuration
    power_group = parser.add_argument_group('Power Configuration')
    power_group.add_argument(
        '--external-power',
        action='store_true',
        help='Use external power supply instead of console'
    )

    # Safety thresholds
    safety_group = parser.add_argument_group('Safety Thresholds')
    safety_group.add_argument(
        '--console-shutoff-temp',
        type=float,
        default=70.0,
        metavar='C',
        help='Console shutoff temperature in Celsius (default: 70.0)'
    )
    safety_group.add_argument(
        '--tx-shutoff-temp',
        type=float,
        default=70.0,
        metavar='C',
        help='TX device shutoff temperature in Celsius (default: 70.0)'
    )
    safety_group.add_argument(
        '--ambient-shutoff-temp',
        type=float,
        default=70.0,
        metavar='C',
        help='Ambient shutoff temperature in Celsius (default: 70.0)'
    )
    safety_group.add_argument(
        '--rapid-temp-shutoff',
        type=float,
        default=40,
        metavar='C',
        help='Rapid temperature shutoff threshold in Celsius (default: 40)'
    )
    safety_group.add_argument(
        '--rapid-temp-time',
        type=float,
        default=5,
        metavar='S',
        help='Time window for rapid temperature shutoff in seconds (default: 5)'
    )
    safety_group.add_argument(
        '--rapid-tx-increase',
        type=float,
        default=3,
        metavar='C',
        help='Rapid TX temperature increase shutoff in Celsius per log interval (default: 3)'
    )
    safety_group.add_argument(
        '--rapid-console-increase',
        type=float,
        default=5,
        metavar='C',
        help='Rapid console temperature increase shutoff in Celsius per log interval (default: 5)'
    )

    # Logging configuration
    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument(
        '--log-interval',
        type=float,
        default=1,
        metavar='S',
        help='Temperature logging interval in seconds (default: 1)'
    )
    log_group.add_argument(
        '-o', '--output-dir',
        type=str,
        default='.',
        metavar='DIR',
        help='Output directory for CSV log files (default: current directory)'
    )

    # Verbosity
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (DEBUG level)'
    )
    verbosity_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress informational messages (WARNING level only)'
    )

    # Automation
    automation_group = parser.add_argument_group('Automation')
    automation_group.add_argument(
        '--no-prompt',
        action='store_true',
        help='Skip user prompts and start immediately (use with caution)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    test = ThermalStressTest(args)

    try:
        test.run()
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down...")
        test.shutdown_event.set()
        test.stop_logging = True
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
