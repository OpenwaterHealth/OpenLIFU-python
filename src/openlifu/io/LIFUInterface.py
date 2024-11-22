import logging

from openlifu.io.LIFUDevice import LIFUDevice
from openlifu.io.LIFUHVController import HVController
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.LIFUUart import LIFUUart

logger = logging.getLogger(__name__)

class LIFUInterface:
    connected: LIFUSignal = LIFUSignal()
    disconnected: LIFUSignal = LIFUSignal()
    data_received: LIFUSignal = LIFUSignal()

    def __init__(self, vid: int = 0x0483, pid: int = 0x57AE, baudrate: int = 921600, timeout: int = 10, test_mode=False) -> None:
        """
        Initialize the LIFUInterface.

        Args:
            vid (int): Vendor ID of the USB device.
            pid (int): Product ID of the USB device.
            baudrate (int): Communication baud rate.
            timeout (int): Read timeout in seconds.
        """
        logger.debug("Initializing LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, pid, baudrate, timeout)

        self.uart = LIFUUart(vid, pid, baudrate, timeout, demo_mode=test_mode)

        # Connect signals to internal handlers
        self.uart.connected.connect(self.connected.emit)
        self.uart.disconnected.connect(self.disconnected.emit)
        self.uart.data_received.connect(self.data_received.emit)

        # Create a LIFUHVController instance as part of the interface
        self.HVController = HVController(self)

        # Create a LIFUDevice instance as part of the interface
        self.Device = LIFUDevice(self)

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
        return self.uart.is_connected()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitoring()
