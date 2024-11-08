from openlifu.io.core import *
from openlifu.io.config import *
import struct
import asyncio
from typing import List

from openlifu.io.i2c_status_packet import I2C_STATUS_Packet
from openlifu.io.i2c_data_packet import I2C_DATA_Packet

class TX7332_IF:
    def __init__(self, afe_interface, identifier: int = -1):
        self.afe_interface = afe_interface
        self.uart = afe_interface._uart
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
            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_WREG, addr=self.afe_interface.i2c_address, reserved=self.identifier, data=data)
        
        self.uart.clear_buffer()
        return response
    
    async def read_register(self, address: int, packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        if packet_id is None:
            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        data = struct.pack('<H', address)

        response = await self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_RREG, addr=self.afe_interface.i2c_address, reserved=self.identifier, data=data)
        self.uart.clear_buffer()

        rUartPacket = UartPacket(buffer=response)
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        if afe_resp.status == 0:
            response = await self.afe_interface.read_data(packet_len=afe_resp.data_len)

        ret_val = 0
        try:
            retPacket = UartPacket(buffer=response)
            data_packet = I2C_DATA_Packet()
            data_packet.from_buffer(buffer=retPacket.data)
            if data_packet.data_len == 4:
                ret_val = struct.unpack('<I', data_packet.pData)[0]

        except Exception as e:
            print("Error reading response:", e)
            
        return ret_val

    async def write_block(self, start_address: int, reg_values: List[int], packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        max_regs_per_block = 62
        num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
        responses = []

        for i in range(num_chunks):
            chunk_start = i * max_regs_per_block
            chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
            chunk = reg_values[chunk_start:chunk_end]

            if packet_id is None:
                self.afe_interface.ctrl_if.packet_count += 1
                packet_id = self.afe_interface.ctrl_if.packet_count

            data_format = '<HBB' + 'I' * len(chunk)
            data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)

            response = await self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_WBLOCK, addr=self.afe_interface.i2c_address, reserved=self.identifier, data=data)
    
            self.uart.clear_buffer()

            responses.append(response)

            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        return responses    

    def print(self):
        print("Controller Instance Information")
        print(f"  Transmitter: {self.identifier}")
