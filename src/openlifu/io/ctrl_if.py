from .core import UART
from .config import OW_CONTROLLER, OW_CMD_PING, OW_CMD_PONG, OW_CMD_ECHO, OW_CMD_TOGGLE_LED, OW_CMD_VERSION, OW_CMD_HWID, OW_CMD_RESET, OW_CTRL_SCAN_I2C, OW_CTRL_SET_SWTRIG, OW_CTRL_GET_SWTRIG, OW_CTRL_START_SWTRIG, OW_CTRL_STOP_SWTRIG, OW_CTRL_SET_HV
from .crc16 import util_crc16
import asyncio
import struct
import json

from .afe_if import AFE_IF

class CTRL_IF:

    _delay = 0.02

    def __init__(self, uart: UART):
        self.uart = uart
        self.packet_count = 0
        self._afe_instances = []

    async def ping(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_PING)
        self.uart.clear_buffer()

        return response

    async def pong(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_PONG)
        self.uart.clear_buffer()

        return response

    async def echo(self, data=None, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_ECHO, data=data)
        self.uart.clear_buffer()
        return response

    async def toggle_led(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_TOGGLE_LED)
        self.uart.clear_buffer()
        return response

    async def version(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_VERSION)
        self.uart.clear_buffer()
        return response

    async def chipid(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_HWID)
        self.uart.clear_buffer()
        return response

    async def reset(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_RESET)
        self.uart.clear_buffer()
        return response

    async def enum_i2c_devices(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_SCAN_I2C)
        self.uart.clear_buffer()

        ret_val = []
        self._afe_instances.clear()

        try:
            retPacket = UartPacket(buffer=response)
            for i in range(retPacket.data_len):
                i2c_address = retPacket.data[i]
                afe_instance = AFE_IF(i2c_address, self)
                self._afe_instances.append(afe_instance)
                ret_val.append(i2c_address)
        except Exception as e:
            print("Error decoding packet:", e)
        return ret_val

    async def set_trigger(self, data=None, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        if data:
            try:
                json_string = json.dumps(data)
            except json.JSONDecodeError as e:
                print(f"Data must be valid JSON: {e}")
                return None

            payload = json_string.encode('utf-8')
        else:
            payload = None

        response = await self.uart.send_ustx(id=1, packetType=OW_CONTROLLER, command=OW_CTRL_SET_SWTRIG, data=payload)
        self.uart.clear_buffer()
        return response

    async def get_trigger(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_GET_SWTRIG, data=None)
        self.uart.clear_buffer()
        data_object = None
        try:
            parsedResp = UartPacket(buffer=response)
            data_object = json.loads(parsedResp.data.decode('utf-8'))
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return data_object

    async def start_trigger(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_START_SWTRIG, data=None)
        self.uart.clear_buffer()

    async def stop_trigger(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_STOP_SWTRIG, data=None)
        self.uart.clear_buffer()

    async def set_hv_supply(self, data=None, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_SET_HV, data=data)
        self.uart.clear_buffer()

    async def get_hv_supply(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_GET_HV, data=None)
        self.uart.clear_buffer()
        return response

    @property
    def afe_devices(self):
        return self._afe_instances

    def print(self):
        print("Controller Instance Information")
        print("  UART Port:")
        self.uart.print()

        print("  AFE_IF Instances:")
        for i in range(len(self._afe_instances)):
            afe_device = self._afe_instances[i]
            afe_device.print()
