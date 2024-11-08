from openlifu.io.core import *
from openlifu.io.config import *
import struct
import asyncio

from openlifu.io.tx7332_if import TX7332_IF
from openlifu.io.i2c_status_packet import I2C_STATUS_Packet

class AFE_IF:
    _delay = 0.02

    def __init__(self, i2c_addr: int, controller):
        self.i2c_addr = i2c_addr
        self.ctrl_if = controller
        self._tx_instances = []
        self._uart = controller.uart
        
    async def ping(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_PING, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
        
    async def pong(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_PONG, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response

    async def echo(self, packet_id=None, data=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_ECHO, addr=self.i2c_addr, data=data)
        self._uart.clear_buffer()
        return response
    
    async def toggle_led(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_TOGGLE_LED, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
    
    async def version(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_VERSION, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
    
    async def chipid(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_HWID, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
        
    async def tx7332_demo(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_DEMO, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
    
    async def reset(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_RESET, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response
    
    async def enum_tx7332_devices(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        self._tx_instances.clear()
        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_AFE_ENUM_TX7332, addr=self.i2c_addr)
        self._uart.clear_buffer()
        rUartPacket = UartPacket(buffer=response)
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        afe_resp.print_packet()
        if afe_resp.status == 0:
            for i in range(afe_resp.reserved):
                self._tx_instances.append(TX7332_IF(self, i))
        return response
    
    async def status(self, packet_id=None):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_STATUS, command=OW_CMD_NOP, addr=self.i2c_addr)
        self._uart.clear_buffer()
        print(response)
        rUartPacket = UartPacket(buffer=response)
        rUartPacket.print_packet()
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        afe_resp.print_packet()

        return response
    
    async def read_data(self, packet_id=None, packet_len: int = 0):
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        await asyncio.sleep(self._delay)
        response = await self._uart.send_ustx(id=packet_id, packetType=OW_AFE_READ, command=packet_len, addr=self.i2c_addr)
        self._uart.clear_buffer()
        return response

    @property
    def tx_devices(self):
        return self._tx_instances
    
    @property
    def i2c_address(self):
        return self.i2c_addr

    def print(self):
        print("  AFE Instance Information")
        formatted_hex = '0x{:02X}'.format(self.i2c_address)
        formatted_hex = formatted_hex.replace(' ', '')  # Remove space
        print(f"    I2C Address: {formatted_hex}")
        print("    Connected TX7332 Devices:")
        for tx_instance in self._tx_instances:
            tx_instance.print()
