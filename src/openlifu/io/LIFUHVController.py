from __future__ import annotations

import logging

from openlifu.io.LIFUConfig import (
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
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

        if self.uart and not self.uart.asyncMode:
            self.uart.check_usb_status()
            if self.uart.is_connected():
                logger.info("HV Console connected.")
            else:
                logger.info("HV Console NOT Connected.")

        # Initialize the high voltage state (should get this from device)
        self.output_voltage = 0.0
        self.is_hv_on = False
        self.is_12v_on = False

        self.supply_voltage = None

    def is_connected(self):
        if self.uart:
            return self.uart.is_connected()

    def ping(self) -> bool:
        """
        Send a ping command to the Console device to verify connectivity.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the ping process.
        """
        try:
            if self.uart.demo_mode:
                return True

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

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

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
            if self.uart.demo_mode:
                return 'v0.1.1'

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_CMD_VERSION
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 3:
                ver = f"v{r.data[0]}.{r.data[1]}.{r.data[2]}"
            else:
                ver = "v0.0.0"
            logger.info(ver)
            return ver

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def echo(self, echo_data=None) -> tuple[bytes, int]:
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
            if self.uart.demo_mode:
                data = b"Hello LIFU!"
                return data, len(data)

            if not self.uart.is_connected():
                raise ValueError("Console Device  not connected")

            # Check if echo_data is a byte array
            if echo_data is not None and not isinstance(echo_data, (bytes, bytearray)):
                raise TypeError("echo_data must be a byte array")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_CMD_ECHO, data=echo_data
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len > 0:
                return r.data, r.data_len
            else:
                return None, None

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def toggle_led(self) -> None:
        """
        Toggle the LED on the Console device.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while toggling the LED.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_CMD_TOGGLE_LED
            )
            self.uart.clear_buffer()
            # r.print_packet()

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

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
            if self.uart.demo_mode:
                return bytes.fromhex("deadbeefcafebabe5566778811223344")

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_POWER, command=OW_CMD_HWID)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 16:
                return r.data.hex()
            else:
                return None

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_12v_off(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning off 12V.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_12V_OFF
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning off 12V")
                return False
            else:
                self.is_12v_on = False
                logger.info("12V turned off successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_12v_on(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning on 12V.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_12V_ON
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning on 12V")
                return False
            else:
                self.is_12v_on = True
                logger.info("12V turned on successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_hv_on(self):
        """
        Turn on the high voltage.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning on high voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_HV_ON
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning on HV Supply")
                return False
            else:
                self.is_hv_on = True
                logger.info("HV Supply turned on successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_hv_off(self):
        """
        Turn off the high voltage.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning off high voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_HV_OFF
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning off HV Supply")
                return False
            else:
                self.is_hv_on = False
                logger.info("HV Supply turned off successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_voltage(self, voltage: float):
        """
        Set the output voltage.

        Args:
            voltage (float): The desired output voltage.

        Raises:
            ValueError: If the controller is not connected or voltage exceeds supply voltage.
        """
        if self.uart.demo_mode:
            return True

        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        # Validate and process the DAC input
        if voltage is None:
            voltage = 0
        elif not (5.0 <= voltage <= 100.0):
            raise ValueError(
                "Voltage input must be within the valid range 5 to 100 Volts)."
            )

        try:
            dac_input = int((voltage / 150) * 4095)
            # logger.info("Setting DAC Value %d.", dac_input)
            # Pack the 12-bit DAC input into two bytes
            data = bytes(
                [
                    (dac_input >> 8) & 0xFF,  # High byte (most significant bits)
                    dac_input & 0xFF,  # Low byte (least significant bits)
                ]
            )

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_SET_HV, data=data
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting HV")
                return False
            else:
                self.supply_voltage = voltage
                logger.info("Output voltage set to %.2fV successfully.", voltage)
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_voltage(self) -> float:
        """
        Get the current output voltage setting.

        Returns:
            float: The current output voltage.

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        try:
            if self.uart.demo_mode:
                return 18.4

            logger.info("Getting current output voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_HV
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting HV")
                return 0.0
            elif r.data_len == 2:
                dac_value = r.data[1] << 8 | r.data[0]
                # logger.info("Got DAC Value %d.", dac_value)
                voltage = dac_value / 4095 * 150
                self.supply_voltage = voltage
                logger.info("Output voltage set to %.2fV successfully.", voltage)
                return voltage
            else:
                logger.error("Error getting output voltage from device")
                return 0.0

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def soft_reset(self) -> bool:
        """
        Perform a soft reset on the Console device.

        Returns:
            bool: True if the reset was successful, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while resetting the device.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console Device  not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_CMD_RESET
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error resetting device")
                return False
            else:
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle
