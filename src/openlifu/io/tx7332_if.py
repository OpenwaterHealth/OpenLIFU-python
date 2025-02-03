import asyncio
import logging
import re
import struct
from typing import List

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


from openlifu.io.config import (
    OW_ERROR,
    OW_TX7332,
    OW_TX7332_RREG,
    OW_TX7332_WBLOCK,
    OW_TX7332_WREG,
)


class TX7332_IF:

    _delay = 0.02

    def __init__(self, ctrl_if, identifier: int = -1):
        self.ctrl_if = ctrl_if
        self.uart = ctrl_if.uart
        self.identifier = identifier

    def get_index(self) -> int:
        return self.identifier

    async def write_register(self, address: int, value: int, packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        data = struct.pack('<HI', address, value)

        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        response = None
        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_WREG, addr=self.identifier, data=data)

        self.uart.clear_buffer()
        return response

    async def read_register(self, address: int, packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        data = struct.pack('<H', address)

        response = None
        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_RREG, addr=self.identifier, data=data)
        self.uart.clear_buffer()

        # rUartPacket = UartPacket(buffer=response)
        # afe_resp = I2C_STATUS_Packet()
        # afe_resp.from_buffer(buffer=rUartPacket.data)
        # if afe_resp.status == 0:
        #     response = await self.afe_interface.read_data(packet_len=afe_resp.data_len)

        ret_val = 0
        # try:
        #     retPacket = UartPacket(buffer=response)
        #     data_packet = I2C_DATA_Packet()
        #     data_packet.from_buffer(buffer=retPacket.data)
        #     if data_packet.data_len == 4:
        #         ret_val = struct.unpack('<I', data_packet.pData)[0]
#
        # except Exception as e:
        #     print("Error reading response:", e)

        return ret_val

    async def write_block(self, start_address: int, reg_values: List[int], packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        max_regs_per_block = 62
        num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
        responses = []
        log.info(f"Write Block Chunks: {num_chunks}")
        for i in range(num_chunks):
            chunk_start = i * max_regs_per_block
            chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
            chunk = reg_values[chunk_start:chunk_end]

            if packet_id is None:
                self.ctrl_if.packet_count += 1
                packet_id = self.ctrl_if.packet_count

            data_format = '<HBB' + 'I' * len(chunk)
            data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)

            await asyncio.sleep(self._delay)
            response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_WBLOCK, addr=self.identifier, data=data)

            self.uart.clear_buffer()

            responses.append(response)

            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        return responses


    def __parse_ti_cfg_file(self, file_path: str) -> list[tuple[str, int, int]]:
        """Parses the given configuration file and extracts all register groups, addresses, and values."""
        parsed_data = []
        pattern = re.compile(r"([\w\d\-]+)\|0x([0-9A-Fa-f]+)\t0x([0-9A-Fa-f]+)")

        with open(file_path) as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    group_name = match.group(1)  # Capture register group name
                    register_address = int(match.group(2), 16)  # Convert hex address to integer
                    register_value = int(match.group(3), 16)  # Convert hex value to integer
                    parsed_data.append((group_name, register_address, register_value))

        return parsed_data

    async def write_ti_config_file(self, file_path:str, packet_id=None) -> bool:
        """
        Reads a TI configuration file and writes the parsed registers to the device.

        :param file_path: Path to the TI config file.
        """
        try:

            if packet_id is None:
                self.ctrl_if.packet_count += 1
                packet_id = self.ctrl_if.packet_count
                

            parsed_registers = self.__parse_ti_cfg_file(file_path)

            for group, addr, value in parsed_registers:
                await asyncio.sleep(self._delay)
        
                print(f"{group:<20}0x{addr:02X}      0x{value:08X}")
                data = struct.pack('<HI', addr, value)
                response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_WREG, addr=self.identifier, data=data)
                if response.packet_type == OW_ERROR:
                    print("Error writing TX device")
                    return False

            return True

        except Exception as e:
            print("Error parsing and writing TI config to TX Device: %s", e)
            raise
        
    def print(self):
        print("Controller Instance Information") # noqa: T201
        print(f"  Transmitter: {self.identifier}") # noqa: T201
