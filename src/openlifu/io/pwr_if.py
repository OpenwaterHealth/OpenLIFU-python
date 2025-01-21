import asyncio

from openlifu.io.config import (
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_PONG,
    OW_CMD_RESET,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_POWER,
    OW_POWER_GET_HV,
    OW_POWER_HV_OFF,
    OW_POWER_HV_ON,
    OW_POWER_SET_HV,
)


class PWR_IF:

    _delay = 0.02

    def __init__(self, ctrl_if):
        self.ctrl_if = ctrl_if
        self.uart = ctrl_if.uart
        self.packet_count = 0

    async def ping(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_PING)
        self.uart.clear_buffer()

        return response

    async def pong(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_PONG)
        self.uart.clear_buffer()

        return response

    async def echo(self, data=None, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_ECHO, data=data)
        self.uart.clear_buffer()
        return response

    async def toggle_led(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_TOGGLE_LED)
        self.uart.clear_buffer()
        return response

    async def version(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_VERSION)
        self.uart.clear_buffer()
        return response

    async def chipid(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_HWID)
        self.uart.clear_buffer()
        return response

    async def reset(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_CMD_RESET)
        self.uart.clear_buffer()
        return response

    async def set_hv_supply(self, packet_id=None, dac_input=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        # Validate and process the DAC input
        if dac_input is not None:
            dac_input = 0
        elif not (0 <= dac_input <= 4095):
            raise ValueError("DAC input must be a 12-bit value (0 to 4095).")

        await asyncio.sleep(self._delay)
        await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_SET_HV, reserved=dac_input)
        self.uart.clear_buffer()

    async def get_hv_supply(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_GET_HV, data=None)
        self.uart.clear_buffer()
        return response

    async def set_hv_supply_on(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_HV_ON)
        self.uart.clear_buffer()

        self.uart.clear_buffer()
        return response

    async def set_hv_supply_off(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_HV_OFF)
        self.uart.clear_buffer()

        self.uart.clear_buffer()
        return response

    @property
    def print(self):
        print("Power Controller Instance Information") # noqa: T201
        print("  UART Port:") # noqa: T201
        self.uart.print()
