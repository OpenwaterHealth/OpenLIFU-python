import json
import serial
import logging
import time
from .config import *
from .utils import util_crc16


# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("UART")
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
        # Placeholder for CRC calculation logic
        # You can replace this with the actual CRC calculation method
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
        """Serialize the UartPacket to a byte buffer."""
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
        print("UartPacket:")
        print("  Packet ID:", self.id)
        print("  Packet Type:", hex(self.packet_type))
        print("  Command:", hex(self.command))
        print("  Address:", hex(self.addr))
        print("  Reserved:", hex(self.reserved))
        print("  Data Length:", self.data_len)
        print("  Data:", self.data.hex())
        print("  CRC:", hex(self.crc))
        
class UART:
    """Handles UART communication."""
    def __init__(self, port: str, baud_rate=921600, timeout=60, align=0):
        """
        Initialize the UART communication.

        :param port: COM port string.
        :param baud_rate: Baud rate for the communication.
        :param timeout: Timeout for the serial communication.
        """
        log.info(f"Connecting to COM port at {port} speed {baud_rate}")
        self.ser = serial.Serial(port, baud_rate, timeout=timeout)
        self.read_buffer = []
        self.align = align

    def close(self):
        """Close the serial connection."""
        self.ser.close()

    def send_ustx(self, id=0, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data = None):
        """Send data over UART."""
        if data:
            if packetType == OW_JSON:
                payload = json.dumps(data).encode('utf-8')
            else:
                payload = data # assume a byte buffer

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
        
        self._tx(packet)
        self._rx()
        return self.read_buffer

    def send(self, buffer):
        """Send a command and wait for a response."""
        self._tx(buffer)
    
    def read(self):
        """Read data until the end byte is encountered."""
        self._rx()
        return self.read_buffer
    
    def _tx(self, data: bytes):
        """Transmit data."""
        try:
            # log.debug(f"TX: {bytes(data).hex()}")
            if self.align > 0:
                while len(data) % self.align != 0:
                    data += bytes([OW_END_BYTE])
            self.ser.write(data)
            self.last_tx=time.monotonic()
        except serial.SerialException as e:
            log.error(f"Error during transmission: {e}")


    def _rx(self):
        """Receive data."""
        try:
            data = self.ser.read_until(bytes([OW_END_BYTE]))
            if data:
                # log.debug(f"RX: {data.hex()}")
                self.read_buffer.extend(data)
                self.last_rx=time.monotonic()
        except serial.SerialException as e:
            log.error(f"Error during reception: {e}")
        
    def clear_buffer(self):
        """Clear the read buffer."""
        self.read_buffer = []

    def print(self):
        print("    Serial Port: ", self.ser.port)
        print("    Serial Baud: ", self.ser.baudrate)
        print("    Serial Parity: ", self.ser.parity)