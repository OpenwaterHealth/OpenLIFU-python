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

    def __init__(self, vid: int = 0x0483, pid: int = 0x57AF, baudrate: int = 921600, timeout: int = 10, test_mode=False) -> None:
        """
        Initialize the LIFUInterface.

        Args:
            vid (int): Vendor ID of the USB device.
            pid (int): Product ID of the USB device.
            baudrate (int): Communication baud rate.
            timeout (int): Read timeout in seconds.
        """
        logger.debug("Initializing LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, pid, baudrate, timeout)

        self.txdevice = TxDevice(uart = LIFUUart(0x0483, 0x57AF, 921600, 10, demo_mode=test_mode))

        # Connect signals to internal handlers
        # self.uart.signal_connect.connect(self.signal_connect.emit)
        # self.uart.signal_disconnect.connect(self.signal_disconnect.emit)
        # self.uart.signal_data_received.connect(self.signal_data_received.emit)
#
        # Create a LIFUHVController instance as part of the interface
        self.hvcontroller = HVController(uart = LIFUUart(0x0483, 0x57A0, 921600, 10, demo_mode=test_mode))

    async def start_monitoring(self, interval: int = 1) -> None:
        """Start monitoring for USB device connections."""
        try:
            await self.uart.monitor_usb_status(interval)
        except Exception as e:
            logger.error("Error starting monitoring: %s", e)


    def stop_monitoring(self) -> None:
        """Stop monitoring for USB device connections."""
        try:
            self.uart.stop_monitoring()
        except Exception as e:
            logger.error("Error stopping monitoring: %s", e)

    def is_device_connected(self) -> bool:
        """
        Check if the device is currently connected.

        Returns:
            bool: True if the device is connected, False otherwise.
        """
        tx_connected = self.txdevice.is_connected()
        hv_connected = self.hvcontroller.is_connected()
        return tx_connected, hv_connected

    def set_solution(self, solution: Solution):
        """
        Load a solution to the device.

        Args:
            solution (Solution): The solution to load.
        """
        try:
            logger.info("Loading solution: %s", solution.name)
            # Convert solution data and send to the device
            self.devicecontroller.set_solution(solution)
            self.hvcontroller.set_voltage(solution.pulse.amplitude)
            logger.info("Solution '%s' loaded successfully.", solution.name)
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
        try:
            logger.info("Retrieving the currently loaded solution.")
            # Example command to request solution metadata and data
            return None
        except Exception as e:
            logger.error("Error retrieving the solution: %s", e)
            raise


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
