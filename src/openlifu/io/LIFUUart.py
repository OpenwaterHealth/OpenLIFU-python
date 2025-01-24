import asyncio
import json
import logging
import threading
import time

import serial
import serial.tools.list_ports

from openlifu.io.config import (
    OW_ACK,
    OW_BAD_PARSE,
    OW_CMD_NOP,
    OW_END_BYTE,
    OW_JSON,
    OW_START_BYTE,
)
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.uartpacket import UartPacket
from openlifu.io.utils import util_crc16

# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("UART")


class LIFUUart:
    def __init__(self, vid, pid, baudrate=921600, timeout=10, align=0, demo_mode=False):
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
        self.asyncMode = False
        self.loop = asyncio.get_event_loop()
        self.monitoring_task = None
        self.demo_mode = demo_mode
        self.read_thread = None
        self.read_buffer = []
        self.last_rx = time.monotonic()
        self.demo_responses = []  # List of predefined responses for testing

        # Signals
        self.signal_connect = LIFUSignal()
        self.signal_disconnect = LIFUSignal()
        self.signal_data_received = LIFUSignal()

    def connect(self):
        """Open the serial port."""
        if self.demo_mode:
            log.info("Demo mode: Simulating UART connection.")
            self.signal_connect.emit("demo_port")
            return
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            log.info("Connected to UART.")
            self.signal_connect.emit(self.port)

            if self.asyncMode:
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
        if self.demo_mode:
            log.info("Demo mode: Simulating UART disconnection.")
            self.signal_disconnect.emit()
            return

        if self.read_thread:
            self.read_thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None
        log.info("Disconnected from UART.")
        self.signal_disconnect.emit()
        self.port = None

    def is_connected(self) -> bool:
        """
        Check if the device is connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        if self.demo_mode:
            return True
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
        if self.demo_mode:
            log.info("Self-monitoring in demo mode.")
            self.connect()
            return
        while True:
            self.check_usb_status()
            await asyncio.sleep(interval)

    def start_monitoring(self, interval=1):
        """Start the periodic USB device connection check."""
        if self.demo_mode:
            log.info("Self-monitoring in demo mode.")
            return
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self.monitor_usb_status(interval))

    def stop_monitoring(self):
        """Stop the periodic USB device connection check."""
        if self.demo_mode:
            log.info("Self-monitoring in demo mode.")
            return
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


    def _read_data(self, timeout=20):
        """Read data from the serial port in a separate thread."""
        if self.demo_mode:
            while self.running:
                if self.demo_responses:
                    data = self.demo_responses.pop(0)
                    log.info("Demo mode: Simulated data received: %s", data)
                    self.signal_data_received.emit(data)
                threading.Event().wait(1000)  # Simulate delay
            return None
        if self.running:
            try:
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    self.read_buffer.extend(data)
                    log.info("Data received: %s", data)
                    self.signal_data_received.emit(data)
            except serial.SerialException as e:
                log.error(f"Serial read error: {e}")
                self.running = False
        else:
            return self.read_packet(timeout=timeout)

    def _tx(self, data: bytes):
        """Send data over UART."""
        if not self.serial or not self.serial.is_open:
            log.error("Serial port is not initialized.")
            return
        if self.demo_mode:
            log.info("Demo mode: Simulating data transmission: %s", data)
            return
        try:
            if self.align > 0:
                while len(data) % self.align != 0:
                    data += bytes([OW_END_BYTE])
            self.serial.write(data)
        except Exception as e:
            log.error(f"Error during transmission: {e}")

    def read_packet(self, timeout=20) -> UartPacket:
        """
        Read a packet from the UART interface.

        This method waits for data to arrive on the serial interface, collects the data,
        and parses it into a UartPacket. If no valid data is received within the timeout,
        a default error packet is returned.

        Returns:
            UartPacket: Parsed packet from the UART interface or an error packet if parsing fails.

        Raises:
            ValueError: If no data is received within the timeout.
        """
        start_time = time.monotonic()
        raw_data = b""
        count = 0

        while timeout == -1 or time.monotonic() - start_time < timeout:
            time.sleep(0.05)  # Wait briefly before retrying
            raw_data += self.serial.read_all()
            if raw_data:  # Break if data is received
                count += 1
                if count > 1:
                    break

        try:
            if not raw_data:
                raise ValueError("No data received from UART within timeout")

            # Attempt to parse the raw data into a UartPacket
            packet = UartPacket(buffer=raw_data)

        except Exception as e:
            # Log the error and create a default error packet
            log.error(f"Error parsing packet: {e}")
            packet = UartPacket(
                id=0,
                packet_type=OW_BAD_PARSE,
                command=0,
                addr=0,
                reserved=0,
                data=[]
            )

        return packet

    def send_packet(self, id=None, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data=None, timeout=20):
        """
        Send a packet over UART.

        Args:
            id (int, optional): Packet ID. If not provided, a unique ID is auto-generated.
            packetType (int): Type of the packet (e.g., OW_ACK, OW_JSON).
            command (int): Command to be sent with the packet.
            addr (int): Address field in the packet.
            reserved (int): Reserved field in the packet.
            data (bytes or dict, optional): Payload data. If packetType is OW_JSON, data is serialized to JSON.
            timeout (in seconds, optional): timeout setting -1 waits forever.

        Returns:
            UartPacket: Parsed response packet if `self.running` is False.
            None: If `self.running` is True or in case of an error.

        Raises:
            ValueError: If data serialization fails or invalid parameters are provided.
        """
        try:
            # Check if serial port is open
            if not self.serial or not self.serial.is_open:
                log.error("Cannot send packet. Serial port is not connected.")
                return None

            # Generate packet ID if not provided
            if id is None:
                self.packet_count += 1
                id = self.packet_count

            # Handle payload
            if data:
                if packetType == OW_JSON:
                    try:
                        payload = json.dumps(data).encode('utf-8')
                    except (TypeError, ValueError) as e:
                        log.error(f"Error serializing data to JSON: {e}")
                        raise ValueError("Invalid data for JSON serialization") from e
                else:
                    if not isinstance(data, (bytes, bytearray)):
                        raise ValueError("Data must be bytes or bytearray if not OW_JSON")
                    payload = data
                payload_length = len(payload)
            else:
                payload_length = 0
                payload = b''

            # Construct packet
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

            # Calculate and append CRC
            crc_value = util_crc16(packet[1:])  # Exclude start byte from CRC
            packet.extend(crc_value.to_bytes(2, 'big'))

            # Append end byte
            packet.append(OW_END_BYTE)

            # Log packet for debugging
            # UartPacket(buffer=packet).print_packet()

            # Transmit packet
            self._tx(packet)

            # If not in running mode, read and return the response packet
            if not self.running:
                return self.read_packet(timeout=timeout)
            else:
                return None

        except ValueError as ve:
            log.error(f"Validation error in send_packet: {ve}")
            raise
        except Exception as e:
            log.error(f"Unexpected error in send_packet: {e}")
            raise

    def clear_buffer(self):
        """Clear the read buffer."""
        self.read_buffer = []

    def run_coroutine(self, coro):
        """Runs a coroutine using the internal event loop."""
        if not self.loop.is_running():
            return self.loop.run_until_complete(coro)
        else:
            return asyncio.create_task(coro)

    def add_demo_response(self, response: bytes):
        """Add a predefined response for demo mode."""
        if self.demo_mode:
            self.demo_responses.append(response)
        else:
            log.warning("Cannot add demo response when not in demo mode.")

    def print(self):
        """Print the current UART configuration."""
        log.info(f"    Serial Port: {self.port}")
        log.info(f"    Serial Baud: {self.baudrate}")
