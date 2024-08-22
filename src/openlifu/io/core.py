import json
import logging
import asyncio
import time
from .config import OW_START_BYTE, OW_END_BYTE, OW_JSON, OW_ACK, OW_CMD_NOP
from .crc16 import util_crc16
from .async_serial import AsyncSerial  # Assuming async_serial.py contains the AsyncSerial class

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
    def __init__(self, port: str, baud_rate=921600, timeout=10, align=0):
        log.info(f"Connecting to COM port at {port} speed {baud_rate}")
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.align = align
        self.serial_port = AsyncSerial(port, baud_rate, timeout)
        self.read_buffer = []

    async def connect(self):
        # Already connected via AsyncSerial's __init__
        pass

    def close(self):
        self.serial_port.close()

    async def send_ustx(self, id=0, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data=None, timeout=10):
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

        await self._tx(packet)
        await self._wait_for_response(timeout)
        return self.read_buffer

    async def send(self, buffer):
        await self._tx(buffer)

    async def read(self):
        await self._rx()
        return self.read_buffer

    async def _tx(self, data: bytes):
        try:
            if self.align > 0:
                while len(data) % self.align != 0:
                    data += bytes([OW_END_BYTE])
            await self.serial_port.write(data)
        except Exception as e:
            log.error(f"Error during transmission: {e}")

    async def _rx(self):
        try:
            while True:
                data = await self.serial_port.read_all()
                if data:
                    self.read_buffer.extend(data)
                    if OW_END_BYTE in data:
                        break
        except Exception as e:
            log.error(f"Error during reception: {e}")

    async def _wait_for_response(self, timeout):
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            await self._rx()
            if self.read_buffer and OW_END_BYTE in self.read_buffer:
                return
            await asyncio.sleep(0.1)
        log.error("Timeout waiting for response")

    def clear_buffer(self):
        self.read_buffer = []

    def print(self):
        print("    Serial Port: ", self.port)
        print("    Serial Baud: ", self.baud_rate)
