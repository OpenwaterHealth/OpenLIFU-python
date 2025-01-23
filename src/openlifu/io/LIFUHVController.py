from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openlifu.io.config import (
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_ERROR,
    OW_POWER,
    OW_POWER_12V_OFF,
    OW_POWER_12V_ON,
)

if TYPE_CHECKING:
    from openlifu.io.LIFUUart import LIFUUart

logger = logging.getLogger(__name__)

class HVController:
    def __init__(self, uart: LIFUUart = None):
        """
        Initialize the HVController.

        Args:
            uart (LIFUUart): The LIFUUart instance for communication.
        """
        self.uart = uart
        self.uart.check_usb_status()
        if self.uart.is_connected():
            logger.info("HV Console connected.")
        else:
            logger.info("HV Console NOT Connected.")

        # Initialize the high voltage state (should get this from device)
        self.output_voltage = 0.0
        self.is_hv_on = False
        self.is_12v_on = False

    def is_connected(self):
        if self.uart:
            return self.uart.is_connected()

    def turn_on(self):
        """
        Turn on the high voltage.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("High voltage controller not connected")

            logger.info("Turning on high voltage.")

            self.is_hv_on = True
            logger.info("High voltage turned on successfully.")
        except Exception as e:
            logger.error("Error turning on high voltage: %s", e)
            raise

    def ping(self) -> bool:
        """
        Send a ping command to the Console device to verify connectivity.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the ping process.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            logger.info("Send Ping to Device.")
            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_PING)
            self.uart.clear_buffer()
            logger.info("Received Ping from Device.")
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error sending ping")
                return False
            else:
                return True

        except Exception as e:
            logger.error("Error Sending Ping: %s", e)
            raise

    def get_version(self) -> str:
        """
        Retrieve the firmware version of the Console device.

        Returns:
            str: Firmware version in the format 'vX.Y.Z'.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while fetching the version.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_VERSION)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 3:
                ver = f'v{r.data[0]}.{r.data[1]}.{r.data[2]}'
            else:
                ver = 'v0.0.0'
            logger.info(ver)
            return ver

        except Exception as e:
            logger.error("Error Toggling LED: %s", e)
            raise

    def echo(self, echo_data = None) -> tuple[bytes, int]:
        """
        Send an echo command to the device with data and receive the same data in response.

        Args:
            echo_data (bytes): The data to send (must be a byte array).

        Returns:
            tuple[bytes, int]: The echoed data and its length.

        Raises:
            ValueError: If the UART is not connected.
            TypeError: If the `echo_data` is not a byte array.
            Exception: If an error occurs during the echo process.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("Console Device  not connected")

            # Check if echo_data is a byte array
            if echo_data is not None and not isinstance(echo_data, (bytes, bytearray)):
                raise TypeError("echo_data must be a byte array")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_ECHO, data=echo_data)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len > 0:
                return r.data, r.data_len
            else:
                return None, None

        except Exception as e:
            logger.error("Error Echo: %s", e)
            raise

    def toggle_led(self) -> None:
        """
        Toggle the LED on the Console device.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while toggling the LED.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_TOGGLE_LED)
            self.uart.clear_buffer()
            # r.print_packet()

        except Exception as e:
            logger.error("Error Toggling LED: %s", e)
            raise

    def get_hardware_id(self) -> str:
        """
        Retrieve the hardware ID of the Console device.

        Returns:
            str: Hardware ID in hexadecimal format.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while retrieving the hardware ID.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_HWID)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 16:
                return r.data.hex()
            else:
                return None

        except Exception as e:
            logger.error("Error Echo: %s", e)
            raise

    def turn_12v_off(self):
        try:
            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning off 12V.")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_POWER_12V_OFF)
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning off 12V")
                return False
            else:
                self.is_12v_on = False
                logger.info("12V turned off successfully.")
                return True

        except Exception as e:
            logger.error("Error turning off 12V: %s", e)
            raise

    def turn_12v_on(self):
        try:
            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning on 12V.")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_POWER_12V_ON)
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning on 12V")
                return False
            else:
                self.is_12v_on = True
                logger.info("12V turned on successfully.")
                return True

        except Exception as e:
            logger.error("Error turning on 12V: %s", e)
            raise

    def turn_off(self):
        """
        Turn off the high voltage.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("High voltage controller not connected")

            logger.info("Turning off high voltage.")
            # Example command to turn off high voltage (adjust packetType and command as needed)

            self.is_hv_on = False
            logger.info("High voltage turned off successfully.")
        except Exception as e:
            logger.error("Error turning off high voltage: %s", e)
            raise

    def set_voltage(self, voltage: float):
            """
            Set the output voltage.

            Args:
                voltage (float): The desired output voltage.

            Raises:
                ValueError: If the controller is not connected or voltage exceeds supply voltage.
            """
            if not self.uart.is_connected():
                raise ValueError("High voltage controller not connected")

            try:
                self.supply_voltage = voltage
                logger.info("Setting output voltage to %.2fV.", voltage)
                # Example command to set the voltage (adjust packetType, command, and format as needed)

                logger.info("Output voltage set to %.2fV successfully.", voltage)
            except Exception as e:
                logger.error("Error setting output voltage: %s", e)
                raise

    def get_voltage(self) -> float:
        """
        Get the current output voltage.

        Returns:
            float: The current output voltage.

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.interface.is_device_connected():
            raise ValueError("High voltage controller not connected")

        try:
            logger.info("Getting current output voltage.")
            # Example command to request voltage reading (adjust packetType and command as needed)

            voltage = self.supply_voltage
            logger.info("Current output voltage: %.2fV.", voltage)
            return voltage
        except Exception as e:
            logger.error("Error getting output voltage: %s", e)
            raise
