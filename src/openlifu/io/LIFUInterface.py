import logging

from openlifu.io.LIFUHVController import HVController
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.LIFUTXDevice import TxDevice
from openlifu.io.LIFUUart import LIFUUart
from openlifu.plan.solution import Solution

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

    def __init__(self, vid: int = 0x0483, tx_pid: int = 0x57AF, con_pid: int = 0x57A0, baudrate: int = 921600, timeout: int = 10, test_mode: bool = False, run_async: bool = False) -> None:

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
        self._test_mode = test_mode
        self._async_mode = run_async

        logger.debug("Initializing TX Module of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, tx_pid, baudrate, timeout)

        self.txdevice = TxDevice(uart = LIFUUart(vid, tx_pid, baudrate, timeout, demo_mode=test_mode, async_mode = run_async))

        # Connect signals to internal handlers
        if self._async_mode:
            self.txdevice.uart.signal_connect.connect(self.signal_connect.emit)
            self.txdevice.uart.signal_disconnect.connect(self.signal_disconnect.emit)
            self.txdevice.uart.signal_data_received.connect(self.signal_data_received.emit)
#
        logger.debug("Initializing Console of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, con_pid, baudrate, timeout)
        # Create a LIFUHVController instance as part of the interface
        self.hvcontroller = HVController(uart = LIFUUart(vid, con_pid, baudrate, timeout, demo_mode=test_mode, async_mode = run_async))

    async def start_monitoring(self, interval: int = 1) -> None:
        """Start monitoring for USB device connections."""
        try:
            await self.txdevice.uart.monitor_usb_status(interval)
        except Exception as e:
            logger.error("Error starting monitoring: %s", e)
            raise e


    def stop_monitoring(self) -> None:
        """Stop monitoring for USB device connections."""
        try:
            self.txdevice.uart.stop_monitoring()
        except Exception as e:
            logger.error("Error stopping monitoring: %s", e)
            raise e

    def is_device_connected(self) -> bool:
        """
        Check if the device is currently connected.

        Returns:
            bool: True if the device is connected, False otherwise.
        """
        tx_connected = self.txdevice.is_connected()
        hv_connected = self.hvcontroller.is_connected()
        return tx_connected, hv_connected

    def set_solution(self, solution: Solution) -> bool:
        """
        Load a solution to the device.

        Args:
            solution (Solution): The solution to load.
        """
        try:
            logger.info("Loading solution: %s", solution.name)
            # Convert solution data and send to the device
            self.txdevice.set_solution(solution)
            self.hvcontroller.set_voltage(solution.pulse.amplitude)
            logger.info("Solution '%s' loaded successfully.", solution.name)
            return True
        except Exception as e:
            logger.error("Error loading solution '%s': %s", solution.name, e)
            raise

    def get_solution(self) -> Solution:
        """
        Retrieve the currently loaded solution from the device.

        Returns:
            Solution: The currently loaded solution.

        Raises:
            ValueError: If no solution is loaded.
        """
        raise NotImplementedError("Parsing of register values on hardware is not implemented yet.")


    def start_sonication(self):
        """
        Start sonication.

        Sets the device to a running state and sends a start command if necessary.
        """
        try:
            logger.info("Start Sonication")
            # Send the solution data to the device
        except Exception as e:
            logger.error("Error Starting sonication: %s", e)
            raise e

    def get_status(self):
        """
        Query the device status.

        Returns:
            int: The device status.
        """
        status = STATUS_ERROR
        return status

    def stop_sonication(self):
        """
        Stop sonication.

        Stops the current sonication process.
        """
        try:
            logger.info("Stop Sonication")
            # Send the solution data to the device
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
