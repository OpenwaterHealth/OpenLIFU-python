import asyncio
import logging
import threading
import time

import serial
import serial.tools.list_ports

from openlifu.io.LIFUSignal import LIFUSignal

# Packet structure constants
OW_START_BYTE = 0xAA
OW_END_BYTE = 0xDD
ID_COUNTER = 0  # Initializing the ID counter

# Packet Types
OW_ACK = 0xE0
OW_NAK = 0xE1
OW_CMD = 0xE2
OW_RESP = 0xE3
OW_DATA = 0xE4
OW_ONE_WIRE = 0xE5
OW_TX7332 = 0xE6
OW_AFE_READ = 0xE7
OW_AFE_SEND = 0xE8
OW_I2C_PASSTHRU = 0xE9
OW_CONTROLLER = 0xEA
OW_POWER = 0xEB
OW_ONEWIRE_RESP = 0xEC
OW_ERROR = 0xEF

OW_SUCCESS = 0x00
OW_UNKNOWN_COMMAND = 0xFC
OW_BAD_CRC = 0xFD
OW_INVALID_PACKET = 0xFE
OW_UNKNOWN_ERROR = 0xFF

# Global Commands
OW_CMD_PING = 0x00
OW_CMD_PONG = 0x01
OW_CMD_VERSION = 0x02
OW_CMD_ECHO = 0x03
OW_CMD_TOGGLE_LED = 0x04
OW_CMD_HWID = 0x05
OW_CMD_GET_TEMP = 0x06
OW_CMD_GET_AMBIENT = 0x07
OW_CMD_DFU = 0x0D
OW_CMD_NOP = 0x0E
OW_CMD_RESET = 0x0F

# Controller Commands
OW_CTRL_SET_SWTRIG = 0x13
OW_CTRL_GET_SWTRIG = 0x14
OW_CTRL_START_SWTRIG = 0x15
OW_CTRL_STOP_SWTRIG = 0x16
OW_CTRL_STATUS_SWTRIG = 0x17
OW_CTRL_RESET = 0x1F

# TX7332 Commands
OW_TX7332_STATUS = 0x20
OW_TX7332_ENUM = 0x21
OW_TX7332_WREG = 0x22
OW_TX7332_RREG = 0x23
OW_TX7332_WBLOCK = 0x24
OW_TX7332_VWREG = 0x25
OW_TX7332_VWBLOCK = 0x26
OW_TX7332_DEMO = 0x2D
OW_TX7332_RESET = 0x2F

# Power Commands
OW_POWER_STATUS = 0x30
OW_POWER_SET_HV = 0x31
OW_POWER_GET_HV = 0x32
OW_POWER_HV_ON = 0x33
OW_POWER_HV_OFF = 0x34
OW_POWER_12V_ON = 0x35
OW_POWER_12V_OFF = 0x36

logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format output with timestamp
)

# Set up logging
log = logging.getLogger("UART")
log.setLevel(logging.DEBUG)

# CRC16-ccitt lookup table
crc16_tab = [
	0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
	0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
	0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
	0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
	0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
	0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
	0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
	0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
	0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
	0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
	0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
	0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
	0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
	0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
	0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
	0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
	0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
	0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
	0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
	0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
	0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
	0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
	0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
	0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
	0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
	0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
	0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
	0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
	0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
	0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
	0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
	0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
]

def util_crc16(buf):
    crc = 0xFFFF

    for byte in buf:
        crc = ((crc << 8) & 0xFFFF) ^ crc16_tab[((crc >> 8) ^ byte) & 0xFF]

    return crc

class UartPacket:
    def __init__(self, id=None, packet_type=None, command=None, addr=None, reserved=None, data=None, buffer=None):
        if buffer:
            self.from_buffer(buffer)
        else:
            self.id = id
            self.packet_type = packet_type
            self.command = command
            self.addr = addr
            self.reserved = reserved
            self.data = data
            self.data_len = len(data)
            self.crc = self.calculate_crc()

    def calculate_crc(self) -> int:
        crc_value = 0xFFFF
        packet = bytearray()
        packet.append(OW_START_BYTE)
        packet.extend(self.id.to_bytes(2, 'big'))
        packet.append(self.packet_type)
        packet.append(self.command)
        packet.append(self.addr)
        packet.append(self.reserved)
        packet.extend(self.data_len.to_bytes(2, 'big'))
        if self.data_len > 0:
            packet.extend(self.data)
        crc_value = util_crc16(packet[1:])
        return crc_value

    def to_bytes(self) -> bytes:
        buffer = bytearray()
        buffer.append(OW_START_BYTE)
        buffer.extend(self.id.to_bytes(2, 'big'))
        buffer.append(self.packet_type)
        buffer.append(self.command)
        buffer.append(self.addr)
        buffer.append(self.reserved)
        buffer.extend(self.data_len.to_bytes(2, 'big'))
        if self.data_len > 0:
            buffer.extend(self.data)
        crc_value = util_crc16(buffer[1:])
        buffer.extend(crc_value.to_bytes(2, 'big'))
        buffer.append(OW_END_BYTE)
        return bytes(buffer)

    def from_buffer(self, buffer: bytes):
        if buffer[0] != OW_START_BYTE or buffer[-1] != OW_END_BYTE:
            raise ValueError("Invalid buffer format")

        self.id = int.from_bytes(buffer[1:3], 'big')
        self.packet_type = buffer[3]
        self.command = buffer[4]
        self.addr = buffer[5]
        self.reserved = buffer[6]
        self.data_len = int.from_bytes(buffer[7:9], 'big')
        self.data = bytearray(buffer[9:9+self.data_len])
        crc_value = util_crc16(buffer[1:9+self.data_len])
        self.crc = int.from_bytes(buffer[9+self.data_len:11+self.data_len], 'big')
        if self.crc != crc_value:
            raise ValueError("CRC mismatch")

    def print_packet(self):
        log.info("UartPacket:")
        log.info(f"  Packet ID: {self.id}")
        log.info(f"  Packet Type: {hex(self.packet_type)}")
        log.info(f"  Command: {hex(self.command)}")
        log.info(f"  Address: {hex(self.addr)}")
        log.info(f"  Reserved: {hex(self.reserved)}")
        log.info(f"  Data Length: {self.data_len}")
        if self.data_len > 0:
            log.info(f"  Data: {self.data.hex()}")
        else:
            log.info("  Data: None")
        log.info(f"  CRC: {hex(self.crc)}")

class LIFUUart:
    def __init__(self, vid, pid, baudrate=921600, timeout=10, align=0, demo_mode=False, async_mode=False):
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
        self.asyncMode = async_mode
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

        if async_mode:
            self.loop = asyncio.get_event_loop()


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
                log.info("starting read thread.")
                self.read_thread = threading.Thread(target=self._read_data)
                self.read_thread.daemon = True
                self.running = True
                self.read_thread.start()
        except serial.SerialException as se:
            log.error(f"Failed to connect to {self.port}: {se}")
            self.running = False
            self.port = None
        except Exception as e:
            raise e

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
            log.debug("Try to connect to device")
            self.port = device
            self.connect()

        elif not device and self.port:
            log.debug("disconnect from device")
            self.running = False
            self.disconnect()

    async def monitor_usb_status(self, interval=1):
        """Periodically check for USB device connection."""
        if self.demo_mode:
            log.debug("Self-monitoring in demo mode.")
            self.connect()
            return
        while True:
            self.check_usb_status()
            await asyncio.sleep(interval)

    def start_monitoring(self, interval=1):
        """Start the periodic USB device connection check."""
        if self.demo_mode:
            log.debug("Self-monitoring in demo mode.")
            return
        if not self.monitoring_task and self.asyncMode:
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
        log.debug("Read Data")
        if self.demo_mode:
            while self.running:
                if self.demo_responses:
                    data = self.demo_responses.pop(0)
                    log.info("Demo mode: Simulated data received: %s", data)
                    self.signal_data_received.emit(data)
                threading.Event().wait(1000)  # Simulate delay
            return None
        if self.asyncMode:
            while self.running:
                try:
                    if self.serial.in_waiting > 0:
                        data = self.serial.read(self.serial.in_waiting)
                        self.read_buffer.extend(data)
                        log.info("Data received: %s", data)
                        self.signal_data_received.emit("Success")
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
            # log.info(f"TX: {len(data)} bytes")
            self.serial.write(data)
        except Exception as e:
            log.error(f"Error during transmission: {e}")
            raise e

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
                packet_type=OW_ERROR,
                command=0,
                addr=0,
                reserved=0,
                data=[]
            )
            raise e

        return packet

    def send_packet(self, id=None, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data=None, timeout=20):
        """
        Send a packet over UART.

        Args:
            id (int, optional): Packet ID. If not provided, a unique ID is auto-generated.
            packetType (int): Type of the packet (e.g., OW_ACK).
            command (int): Command to be sent with the packet.
            addr (int): Address field in the packet.
            reserved (int): Reserved field in the packet.
            data (bytes or dict, optional): Payload data.
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
                if not isinstance(data, (bytes, bytearray)):
                    raise ValueError("Data must be bytes or bytearray")
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
