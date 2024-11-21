import logging
from typing import Optional

from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.LIFUUart import LIFUUart

logger = logging.getLogger(__name__)

class LIFUInterface:
    connected: LIFUSignal = LIFUSignal()
    disconnected: LIFUSignal = LIFUSignal()
    data_received: LIFUSignal = LIFUSignal()

    def __init__(self, vid: int = 0x0483, pid: int = 0x57AE, baudrate: int = 921600, timeout: int = 10) -> None:
        """
        Initialize the LIFUInterface.

        Args:
            vid (int): Vendor ID of the USB device.
            pid (int): Product ID of the USB device.
            baudrate (int): Communication baud rate.
            timeout (int): Read timeout in seconds.
        """
        self.uart = LIFUUart(vid, pid, baudrate, timeout)
        self.uart.connected.connect(self.connected.emit)
        self.uart.disconnected.connect(self.disconnected.emit)
        self.uart.data_received.connect(self.data_received.emit)

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

    def send_ping(self) -> None:
        """Send a ping command over UART."""
        try:
            self.uart.run_coroutine(self.uart.send_packet(packetType=0xE2, command=0x00))
        except Exception as e:
            logger.error("Error sending ping: %s", e)

    def toggle_treatment_run(self, capture_on: bool) -> None:
        """Toggle the treatment run state."""
        command = 0x07 if capture_on else 0x06
        try:
            self.uart.run_coroutine(self.uart.send_packet(packetType=0xE2, command=command))
        except Exception as e:
            logger.error("Error toggling treatment run: %s", e)

    def send_custom_packet(self, packetType: int, command: int, data: Optional[bytes] = None) -> None:
        """Send a custom packet over UART."""
        try:
            self.uart.run_coroutine(self.uart.send_packet(packetType=packetType, command=command, data=data))
        except Exception as e:
            logger.error("Error sending custom packet: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitoring()
