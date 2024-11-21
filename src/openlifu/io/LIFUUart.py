import asyncio
import json
import logging
import threading

import serial
import serial.tools.list_ports

from openlifu.io.config import OW_ACK, OW_CMD_NOP, OW_END_BYTE, OW_JSON, OW_START_BYTE
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.uartpacket import UartPacket
from openlifu.io.utils import util_crc16

# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("UART")


class LIFUUart:
    def __init__(self, vid, pid, baudrate=921600, timeout=10, align=0):
        """
        Initialize the UART instance.

        Args:
            vid (int): Vendor ID of the USB device.
            pid (int): Product ID of the USB device.
            baudrate (int): Communication speed.
            timeout (int): Read timeout in seconds.
            align (int): Data alignment parameter.
        """
        self.vid = vid
        self.pid = pid
        self.port = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.align = align
        self.packet_count = 0
        self.serial = None
        self.running = False
        self.read_thread = None
        self.read_buffer = []
        self.loop = asyncio.get_event_loop()
        self.monitoring_task = None

        # Signals
        self.connected = LIFUSignal()
        self.disconnected = LIFUSignal()
        self.data_received = LIFUSignal()

    def connect(self):
        """Open the serial port."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.running = True
            log.info("Connected to UART.")
            self.connected.emit(self.port)

            # Start the reading thread
            self.read_thread = threading.Thread(target=self._read_data)
            self.read_thread.daemon = True
            self.read_thread.start()
        except Exception as e:
            log.error(f"Failed to connect to {self.port}: {e}")
            self.running = False
            self.port = None

    def disconnect(self):
        """Close the serial port."""
        self.running = False
        if self.read_thread:
            self.read_thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None
        log.info("Disconnected from UART.")
        self.disconnected.emit()
        self.port = None

    def is_connected(self) -> bool:
        """
        Check if the device is connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.port is not None and self.serial is not None and self.serial.is_open

    def check_usb_status(self):
        """Check if the USB device is connected or disconnected."""
        device = self.list_vcp_with_vid_pid()
        if device and not self.port:
            self.port = device
            self.connect()
        elif not device and self.port:
            self.disconnect()

    async def monitor_usb_status(self, interval=1):
        """Periodically check for USB device connection."""
        while True:
            self.check_usb_status()
            await asyncio.sleep(interval)

    def start_monitoring(self, interval=1):
        """Start the periodic USB device connection check."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self.monitor_usb_status(interval))

    def stop_monitoring(self):
        """Stop the periodic USB device connection check."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None

    def list_vcp_with_vid_pid(self):
        """Find the USB device by VID and PID."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if hasattr(port, 'vid') and hasattr(port, 'pid') and port.vid == self.vid and port.pid == self.pid:
                return port.device
        return None


    def _read_data(self):
        """Read data from the serial port in a separate thread."""
        while self.running:
            try:
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    self.read_buffer.extend(data)
                    log.info("Data received: %s", data)
                    self.data_received.emit(data)
            except serial.SerialException as e:
                log.error(f"Serial read error: {e}")
                self.running = False

    def _tx(self, data: bytes):
        """Send data over UART."""
        if not self.serial or not self.serial.is_open:
            log.error("Serial port is not initialized.")
            return
        try:
            if self.align > 0:
                while len(data) % self.align != 0:
                    data += bytes([OW_END_BYTE])
            self.serial.write(data)
        except Exception as e:
            log.error(f"Error during transmission: {e}")

    async def send_packet(self, id=None, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data=None):
        """Send a packet over UART."""
        if not self.serial or not self.serial.is_open:
            log.error("Cannot send packet. Serial port is not connected.")
            return

        if id is None:
            self.packet_count += 1
            id = self.packet_count

        if data:
            if packetType == OW_JSON:
                payload = json.dumps(data).encode('utf-8')
            else:
                payload = data
            payload_length = len(payload)
        else:
            payload_length = 0

        packet = bytearray()
        packet.append(OW_START_BYTE)
        packet.extend(id.to_bytes(2, 'big'))
        packet.append(packetType)
        packet.append(command)
        packet.append(addr)
        packet.append(reserved)
        packet.extend(payload_length.to_bytes(2, 'big'))
        if payload_length > 0:
            packet.extend(payload)
        crc_value = util_crc16(packet[1:])
        packet.extend(crc_value.to_bytes(2, 'big'))
        packet.append(OW_END_BYTE)
        UartPacket(buffer=packet).print_packet()

        self._tx(packet)

    def clear_buffer(self):
        """Clear the read buffer."""
        self.read_buffer = []

    def run_coroutine(self, coro):
        """Runs a coroutine using the internal event loop."""
        if not self.loop.is_running():
            return self.loop.run_until_complete(coro)
        else:
            return asyncio.create_task(coro)

    def print(self):
        """Print the current UART configuration."""
        log.info(f"    Serial Port: {self.port}")
        log.info(f"    Serial Baud: {self.baudrate}")
