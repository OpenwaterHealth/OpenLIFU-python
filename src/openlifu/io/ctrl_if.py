import asyncio
import json
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from openlifu.io.config import (
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_PONG,
    OW_CMD_RESET,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_CONTROLLER,
    OW_CTRL_GET_HV,
    OW_CTRL_GET_SWTRIG,
    OW_CTRL_SET_HV,
    OW_CTRL_SET_SWTRIG,
    OW_CTRL_START_SWTRIG,
    OW_CTRL_STOP_SWTRIG,
    OW_TX7332,
    OW_TX7332_DEMO,
    OW_TX7332_ENUM,
)
from openlifu.io.core import UART, UartPacket
from openlifu.io.tx7332_if import TX7332_IF


class CTRL_IF:

    _delay = 0.02

    def __init__(self, uart: UART):
        self.uart = uart
        self.packet_count = 0
        self._tx_instances = []

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

    async def enum_tx7332_devices(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        self._tx_instances.clear()
        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_ENUM)

        rUartPacket = UartPacket(buffer=response)
        if rUartPacket.reserved > 0:
            for i in range(rUartPacket.reserved):
                self._tx_instances.append(TX7332_IF(self, i))


        return self._tx_instances

    async def tx7332_demo(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_TX7332, command=OW_TX7332_DEMO)

        return response

    async def enum_i2c_devices(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        # response = await self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_SCAN_I2C)
        self.uart.clear_buffer()

        ret_val = []
        return ret_val

    async def set_trigger(self, data=None, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        if data:
            try:
                json_string = json.dumps(data)
            except json.JSONDecodeError as e:
                log.info(f"Data must be valid JSON: {e}")
                return None

            payload = json_string.encode('utf-8')
        else:
            payload = None

        await asyncio.sleep(self._delay)
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
            log.info(f"Error decoding JSON: {e}")
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
    def tx_devices(self):
        return self._tx_instances

    @property
    def print(self):
        print("Controller Instance Information")  # noqa: T201
        print("  UART Port:")  # noqa: T201
        self.uart.print()
