import asyncio

from openlifu.io.config import (
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_PONG,
    OW_CMD_RESET,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_ERROR,
    OW_POWER,
    OW_POWER_12V_OFF,
    OW_POWER_12V_ON,
    OW_POWER_GET_HV,
    OW_POWER_HV_OFF,
    OW_POWER_HV_ON,
    OW_POWER_SET_HV,
)
from openlifu.io.core import UART


class PWR_IF:

    _delay = 0.02

    def __init__(self, uart: UART):
        self.uart = uart
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
        if dac_input is None:
            dac_input = 0
        elif not (0 <= dac_input <= 4095):
            raise ValueError("DAC input must be a 12-bit value (0 to 4095).")

        # Pack the 12-bit DAC input into two bytes
        data = [
            (dac_input >> 8) & 0xFF,  # High byte (most significant bits)
            dac_input & 0xFF          # Low byte (least significant bits)
        ]
        await asyncio.sleep(self._delay)
        await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_SET_HV, data=data)
        self.uart.clear_buffer()

    async def set_voltage(self, voltage: float, packet_id=None, ):
        # Validate and process the DAC input
        if voltage is None:
            voltage = 0
        elif not (5.0<= voltage <= 100.0):
            raise ValueError("Voltage input must be within the valid range 5 to 100 Volts).")

        try:
            dac_input = int((voltage / 150) * 4095)
            # logger.info("Setting DAC Value %d.", dac_input)
            # Pack the 12-bit DAC input into two bytes
            data = bytes([
                (dac_input >> 8) & 0xFF,  # High byte (most significant bits)
                dac_input & 0xFF          # Low byte (least significant bits)
            ])

            await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_SET_HV, data=data)
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                return False
            else:
                self.supply_voltage = voltage
                print("Output voltage set to %.2fV successfully.", voltage)
                return True

        except Exception as e:
            print("Error setting output voltage: %s", e)
            raise

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
        return response

    async def set_hv_supply_off(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_HV_OFF)
        self.uart.clear_buffer()
        return response


    async def set_12v_on(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_12V_ON)
        self.uart.clear_buffer()
        return response

    async def set_12v_off(self, packet_id=None):
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        await asyncio.sleep(self._delay)
        response = await self.uart.send_ustx(id=packet_id, packetType=OW_POWER, command=OW_POWER_12V_OFF)
        self.uart.clear_buffer()
        return response

    @property
    def print(self):
        print("Power Controller Instance Information") # noqa: T201
        print("  UART Port:") # noqa: T201
        self.uart.print()
