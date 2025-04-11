from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Dict

from openlifu.io.LIFUHVController import HVController
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.LIFUTXDevice import TxDevice
from openlifu.io.LIFUUart import LIFUUart
from openlifu.plan.solution import Solution


class LIFUInterfaceStatus(Enum):
    STATUS_COMMS_ERROR = -1
    STATUS_SYS_OFF = 0
    STATUS_SYS_POWERUP = 1
    STATUS_SYS_ON = 2
    STATUS_PROGRAMMING = 3
    STATUS_READY = 4
    STATUS_NOT_READY = 5
    STATUS_RUNNING = 6
    STATUS_FINISHED = 7
    STATUS_ERROR = 8

logger = logging.getLogger(__name__)

class LIFUInterface:
    signal_connect: LIFUSignal = LIFUSignal()
    signal_disconnect: LIFUSignal = LIFUSignal()
    signal_data_received: LIFUSignal = LIFUSignal()
    hvcontroller: HVController = None
    txdevice: TxDevice = None

    def __init__(self, vid: int = 0x0483, tx_pid: int = 0x57AF, con_pid: int = 0x57A0, baudrate: int = 921600, timeout: int = 10, TX_test_mode: bool = False, HV_test_mode: bool = False, run_async: bool = False) -> None:
        """
        Initialize the LIFUInterface with given parameters and store them in the class.

        Args:
            vid (int): Vendor ID of the USB device.
            tx_pid (int): Product ID for TX device.
            con_pid (int): Product ID for console device.
            baudrate (int): Communication baud rate.
            timeout (int): Read timeout in seconds.
            test_mode (bool): Enable test mode.
            run_async (bool): Enable asynchronous operation.
        """
        # Store parameters in instance variables
        self.vid = vid
        self.tx_pid = tx_pid
        self.con_pid = con_pid
        self.baudrate = baudrate
        self.timeout = timeout
        self._test_mode = TX_test_mode
        self._async_mode = run_async
        self._tx_uart = None
        self._hv_uart = None

        # Create a TXDevice instance as part of the interface
        logger.debug("Initializing TX Module of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, tx_pid, baudrate, timeout)
        self._tx_uart = LIFUUart(vid=vid, pid=tx_pid, baudrate=baudrate, timeout=timeout, desc="TX", demo_mode=TX_test_mode, async_mode=run_async)
        self.txdevice = TxDevice(uart=self._tx_uart)

        # Create a LIFUHVController instance as part of the interface
        logger.debug("Initializing Console of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, con_pid, baudrate, timeout)
        self._hv_uart = LIFUUart(vid=vid, pid=con_pid, baudrate=baudrate, timeout=timeout, desc="HV", demo_mode=HV_test_mode, async_mode=run_async)
        self.hvcontroller = HVController(uart=self._hv_uart)

        # Connect signals to internal handlers
        if self._async_mode:
            self._tx_uart.signal_connect.connect(self.signal_connect.emit)
            self._tx_uart.signal_disconnect.connect(self.signal_disconnect.emit)
            self._tx_uart.signal_data_received.connect(self.signal_data_received.emit)
            self._hv_uart.signal_connect.connect(self.signal_connect.emit)
            self._hv_uart.signal_disconnect.connect(self.signal_disconnect.emit)
            self._hv_uart.signal_data_received.connect(self.signal_data_received.emit)

    async def start_monitoring(self, interval: int = 1) -> None:
        """Start monitoring for USB device connections."""
        try:
            await asyncio.gather(
                self._tx_uart.monitor_usb_status(interval),
                self._hv_uart.monitor_usb_status(interval)
            )

        except Exception as e:
            logger.error("Error starting monitoring: %s", e)
            raise e

    def stop_monitoring(self) -> None:
        """Stop monitoring for USB device connections."""
        try:
            self._tx_uart.stop_monitoring()
            self._hv_uart.stop_monitoring()
        except Exception as e:
            logger.error("Error stopping monitoring: %s", e)
            raise e

    def is_device_connected(self) -> tuple:
        """
        Check if the device is currently connected.

        Returns:
            tuple: (tx_connected, hv_connected)
        """
        tx_connected = self.txdevice.is_connected()
        hv_connected = self.hvcontroller.is_connected()
        return tx_connected, hv_connected

    def set_solution(self,
                     solution: Solution | Dict,
                     profile_index:int=1,
                     profile_increment:bool=True) -> bool:
        """
        Load a solution to the device.

        Args:
            solution (Solution): The solution to load.
            profile_index (int): The profile index to load the solution to (defaults to 0)
            profile_increment (bool): Increment the profile index
        """
        try:
            #if self._test_mode:
            #    return True

            if isinstance(solution, Solution):
                solution = solution.to_dict()

            if "name" in solution:
                solution_name = solution["name"]
                solution_name = f'Solution "{solution_name}"'
            else:
                solution_name = "Solution"

            voltage = solution['pulse']['amplitude']

            logger.info("Loading %s...", solution_name)
            # Convert solution data and send to the device
            self.txdevice.set_solution(
                    pulse = solution['pulse'],
                    delays = solution['delays'],
                    apodizations= solution['apodizations'],
                    sequence= solution['sequence'],
                    profile_index=profile_index,
                    profile_increment=profile_increment
                )
            self.hvcontroller.set_voltage(voltage)
            logger.info("%s loaded successfully.", solution_name)
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle
        except Exception as e:
            logger.error("Error loading %s: %s", solution_name, e)
            raise

    def start_sonication(self) -> bool:
        """
        Start sonication.

        Sets the device to a running state and sends a start command if necessary.
        """
        try:
            if self._test_mode:
                return True

            logger.info("Turn ON HV")
            bHvOn = self.hvcontroller.turn_hv_on()

            logger.info("Start Sonication")
            # Send the solution data to the device
            bTriggerOn = self.txdevice.start_trigger()

            if bTriggerOn and bHvOn:
                logger.info("Sonication started successfully.")
                return True
            else:
                logger.error("Failed to start sonication.")
                return False

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Error Starting sonication: %s", e)
            raise e

    def get_status(self):
        """
        Query the device status.

        Returns:
            int: The device status.
        """
        if self._test_mode:
            return LIFUInterfaceStatus.STATUS_READY

        status = LIFUInterfaceStatus.STATUS_ERROR
        return status

    def stop_sonication(self) -> bool:
        """
        Stop sonication.

        Stops the current sonication process.
        """
        try:
            if self._test_mode:
                return True

            logger.info("Stop Sonication")
            # Send the solution data to the device
            bTriggerOff = self.txdevice.stop_trigger()
            bHvOff = self.hvcontroller.turn_hv_off()

            if bTriggerOff and bHvOff:
                logger.info("Sonication stopped successfully.")
                return True
            else:
                logger.error("Failed to stop sonication.")
                return False

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Error Stopping sonication: %s", e)
            raise e

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitoring()
        if self.txdevice:
            self.txdevice.disconnect()
        if self.hvcontroller:
            self.hvcontroller.disconnect()
