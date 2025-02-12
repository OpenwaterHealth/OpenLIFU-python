from __future__ import annotations

import json
import logging
import re
import struct
from typing import TYPE_CHECKING, List

from openlifu.io.config import (
    OW_CMD_ECHO,
    OW_CMD_GET_TEMP,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_RESET,
    OW_CMD_TEST,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_CONTROLLER,
    OW_CTRL_GET_SWTRIG,
    OW_CTRL_SET_SWTRIG,
    OW_CTRL_START_SWTRIG,
    OW_CTRL_STOP_SWTRIG,
    OW_ERROR,
    OW_TX7332,
    OW_TX7332_DEMO,
    OW_TX7332_ENUM,
    OW_TX7332_RREG,
    OW_TX7332_VWBLOCK,
    OW_TX7332_VWREG,
    OW_TX7332_WBLOCK,
    OW_TX7332_WREG,
)
from openlifu.io.LIFUUart import LIFUUart
from openlifu.io.tx7332_if import TX7332_IF

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

class TxDevice:
    def __init__(self, uart: LIFUUart = None):
        """
        Initialize the TxDevice.

        Args:
            uart (LIFUUart): The LIFUUart instance for communication.
        """
        self._tx_instances = []
        self.uart = uart
        self.uart.check_usb_status()
        if self.uart.is_connected():
            logger.info("TX Device connected.")
        else:
            logger.info("TX Device NOT Connected.")

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


    def is_connected(self) -> bool:
        """
        Check if the TX device is connected.

        Returns:
            bool: True if the device is connected, False otherwise.
        """
        if self.uart:
            return self.uart.is_connected()

    def ping(self) -> bool:
        """
        Send a ping command to the TX device to verify connectivity.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the ping process.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            logger.info("Send Ping to Device.")
            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_PING)
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
        Retrieve the firmware version of the TX device.

        Returns:
            str: Firmware version in the format 'vX.Y.Z'.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while fetching the version.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_VERSION)
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
                raise ValueError("TX Device not connected")

            # Check if echo_data is a byte array
            if echo_data is not None and not isinstance(echo_data, (bytes, bytearray)):
                raise TypeError("echo_data must be a byte array")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_ECHO, data=echo_data)
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
        Toggle the LED on the TX device.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while toggling the LED.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_TOGGLE_LED)
            self.uart.clear_buffer()
            # r.print_packet()

        except Exception as e:
            logger.error("Error Toggling LED: %s", e)
            raise

    def get_hardware_id(self) -> str:
        """
        Retrieve the hardware ID of the TX device.

        Returns:
            str: Hardware ID in hexadecimal format.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while retrieving the hardware ID.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_HWID)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 16:
                return r.data.hex()
            else:
                return None

        except Exception as e:
            logger.error("Error Echo: %s", e)
            raise

    def get_temperature(self) -> float:
        """
        Retrieve the temperature reading from the TX device.

        Returns:
            float: Temperature value in Celsius.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs or the received data length is invalid.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Send the GET_TEMP command
            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_GET_TEMP)
            self.uart.clear_buffer()
            # r.print_packet()

            # Check if the data length matches a float (4 bytes)
            if r.data_len == 4:
                # Unpack the float value from the received data (assuming little-endian)
                temperature = struct.unpack('<f', r.data)[0]
                return temperature
            else:
                raise ValueError("Invalid data length received for temperature")

        except Exception as e:
            logger.error("Error retrieving temperature: %s", e)
            raise

    def set_trigger(self, data=None) -> dict:
        """
        Set the trigger configuration on the TX device.

        Args:
            data (dict): A dictionary containing the trigger configuration.

        Returns:
            dict: JSON response from the device.

        Raises:
            ValueError: If `data` is None or the UART is not connected.
            Exception: If an error occurs while setting the trigger.
        """
        try:
            # Ensure data is not None and is a valid dictionary
            if data is None:
                raise ValueError("Data cannot be None.")

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            try:
                json_string = json.dumps(data)
            except json.JSONDecodeError as e:
                logger.error(f"Data must be valid JSON: {e}")
                return None

            payload = json_string.encode('utf-8')

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_SET_SWTRIG, data=payload)
            self.uart.clear_buffer()

            if r.packet_type != OW_ERROR and r.data_len > 0:
                # Parse response as JSON, if possible
                try:
                    response_json = json.loads(r.data.decode('utf-8'))
                    return response_json
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
                    return None
            else:
                return None

        except Exception as e:
            logger.error("Error Enumerating TX Devices: %s", e)
            raise

    def get_trigger(self) -> bool:
        """
        Start the trigger on the TX device.

        Returns:
            bool: True if the trigger was started successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while starting the trigger.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_GET_SWTRIG, data=None)
            self.uart.clear_buffer()
            data_object = None
            try:
                data_object = json.loads(r.data.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
            return data_object

        except Exception as e:
            logger.error("Error Enumerating TX Devices: %s", e)
            raise

    def start_trigger(self) -> bool:
        """
        Start the trigger on the TX device.

        Returns:
            bool: True if the trigger was started successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while starting the trigger.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_START_SWTRIG, data=None)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error starting trigger")
                return False
            else:
                return True

        except Exception as e:
            logger.error("Error Enumerating TX Devices: %s", e)
            raise

    def stop_trigger(self) -> bool:
        """
        Stop the trigger on the TX device.

        This method sends a command to stop the software trigger on the TX device.
        It checks the device's connection status and handles errors appropriately.

        Returns:
            bool: True if the trigger was successfully stopped, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the operation.
        """
        try:
            # Check if the device is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Send the STOP_SWTRIG command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_CONTROLLER,
                command=OW_CTRL_STOP_SWTRIG,
                data=None
            )

            # Clear the UART buffer to prepare for further communication
            self.uart.clear_buffer()

            # Log the received packet for debugging purposes
            # r.print_packet()

            # Check the packet type to determine success
            if r.packet_type == OW_ERROR:
                logger.error("Error stopping trigger")
                return False
            else:
                return True

        except Exception as e:
            # Log any exceptions that occur and re-raise them
            logger.error("Error stopping trigger: %s", e)
            raise

    def soft_reset(self) -> bool:
        """
        Perform a soft reset on the TX device.

        Returns:
            bool: True if the reset was successful, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while resetting the device.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_RESET)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error resetting device")
                return False
            else:
                return True

        except Exception as e:
            logger.error("Error Trying to reset TX Device: %s", e)
            raise

    def run_test(self) -> bool:
        """
        Perform a self test on the TX device.

        Returns:
            bool: True if the test was successful, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while test was performed on the device.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_TEST)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error running test on device")
                return False
            else:
                return True

        except Exception as e:
            logger.error("Error running test on TX Devices: %s", e)
            raise

    def enum_tx7332_devices(self) -> list[TX7332_IF]:
        """
        Enumerate TX7332 devices connected to the TX device.

        Returns:
            list[TX7332_IF]: A list of TX7332 interface instances.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during enumeration.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            self._tx_instances.clear()
            r = self.uart.send_packet(id=None, packetType=OW_TX7332, command=OW_TX7332_ENUM)
            self.uart.clear_buffer()
            # r.print_packet()
            # Clear the array before appending
            self._tx_instances.clear()
            if r.packet_type != OW_ERROR and r.reserved > 0:
                for i in range(r.reserved):
                    self._tx_instances.append(TX7332_IF(self, i))
            else:
                logger.info("Error enumerating TX devices.")

            logger.info("TX Device Count: %d", len(self._tx_instances))
            return self._tx_instances

        except Exception as e:
            logger.error("Error Enumerating TX Devices: %s", e)
            raise

    def demo_tx7332(self) -> bool:
        """
        Sets all TX7332 chip registers with a test waveform.

        Returns:
            bool: True if all chips are programmed successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_TX7332, command=OW_TX7332_DEMO)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error demoing TX devices")
                return False

            return True

        except Exception as e:
            logger.error("Error Enumerating TX Devices: %s", e)
            raise

    def write_register(self, identifier:int, address: int, value: int) -> bool:
        """
        Write a value to a register in the TX device.

        Args:
            address (int): The register address to write to.
            value (int): The value to write to the register.

        Returns:
            bool: True if the write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, or the identifier is invalid.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            # Check if the UART is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if identifier < 0:
                raise ValueError("TX Chip address NOT SET")
            if identifier > 1:
                raise ValueError("TX Chip address must be in the range 0-1")

            # Pack the address and value into the required format
            try:
                data = struct.pack('<HI', address, value)
            except struct.error as e:
                logger.error(f"Error packing address and value: {e}")
                raise ValueError("Invalid address or value format") from e

            # Send the write command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_WREG,
                addr=identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()

            # Check the response for errors
            if r.packet_type == OW_ERROR:
                logger.error("Error writing TX register value")
                return False

            logger.info(f"Successfully wrote value 0x{value:08X} to register 0x{address:04X}")
            return True

        except ValueError as ve:
            logger.error(f"Validation error in write_register: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in write_register: {e}")
            raise

    def read_register(self, address: int) -> int:
        """
        Read a register value from the TX device.

        Args:
            address (int): The register address to read.

        Returns:
            int: The value of the register if successful, or 0 on failure.

        Raises:
            ValueError: If the identifier is not set or is out of range.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if self.identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            if self.identifier > 1:
                raise ValueError("TX Chip address must be in the range 0-1")

            # Pack the address into the required format
            try:
                data = struct.pack('<H', address)
            except struct.error as e:
                logger.error(f"Error packing address {address}: {e}")
                raise ValueError("Invalid address format") from e

            # Send the read command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_RREG,
                addr=self.identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()

            # Check for errors in the response
            if r.packet_type == OW_ERROR:
                logger.error("Error reading TX register value")
                return 0

            # Verify data length and unpack the register value
            if r.data_len == 4:
                try:
                    return struct.unpack('<I', r.data)[0]
                except struct.error as e:
                    logger.error(f"Error unpacking register value: {e}")
                    return 0
            else:
                logger.error(f"Unexpected data length: {r.data_len}")
                return 0

        except ValueError as ve:
            logger.error(f"Validation error in read_register: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in read_register: {e}")
            raise

    def write_block(self, identifier: int, start_address: int, reg_values: List[int]) -> bool:
        """
        Write a block of register values to the TX device.

        Args:
            start_address (int): The starting register address to write to.
            reg_values (List[int]): List of register values to write.

        Returns:
            bool: True if the block write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, the identifier is invalid, or parameters are out of range.
        """
        try:
            # Ensure the UART connection is active
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if identifier < 0:
                raise ValueError("TX Chip address NOT SET")
            if identifier > 1:
                raise ValueError("TX Chip address must be in the range 0-1")

            # Validate the reg_values list
            if not reg_values or not isinstance(reg_values, list):
                raise ValueError("Invalid register values: Must be a non-empty list of integers")
            if any(not isinstance(value, int) for value in reg_values):
                raise ValueError("Invalid register values: All elements must be integers")

            # Configure chunking for large blocks
            max_regs_per_block = 62  # Maximum registers per block due to payload size
            num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
            logger.info(f"Write Block: Total chunks = {num_chunks}")

            # Write each chunk
            for i in range(num_chunks):
                chunk_start = i * max_regs_per_block
                chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
                chunk = reg_values[chunk_start:chunk_end]

                # Pack the chunk into the required data format
                try:
                    data_format = '<HBB' + 'I' * len(chunk)  # Start address (H), chunk length (B), reserved (B), values (I...)
                    data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)
                except struct.error as e:
                    logger.error(f"Error packing data for chunk {i}: {e}")
                    return False

                # Send the packet
                r = self.uart.send_packet(
                    id=None,
                    packetType=OW_TX7332,
                    command=OW_TX7332_WBLOCK,
                    addr=identifier,
                    data=data
                )

                # Clear the UART buffer after sending
                self.uart.clear_buffer()

                # Check for errors in the response
                if r.packet_type == OW_ERROR:
                    logger.error(f"Error writing TX block at chunk {i}")
                    return False

            logger.info("Block write successful")
            return True

        except ValueError as ve:
            logger.error(f"Validation error in write_block: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in write_block: {e}")
            raise

    def write_register_verify(self, address: int, value: int) -> bool:
        """
        Write a value to a register in the TX device with verification.

        Args:
            address (int): The register address to write to.
            value (int): The value to write to the register.

        Returns:
            bool: True if the write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, or the identifier is invalid.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            # Check if the UART is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if self.identifier < 0:
                raise ValueError("TX Chip address NOT SET")
            if self.identifier > 1:
                raise ValueError("TX Chip address must be in the range 0-1")

            # Pack the address and value into the required format
            try:
                data = struct.pack('<HI', address, value)
            except struct.error as e:
                logger.error(f"Error packing address and value: {e}")
                raise ValueError("Invalid address or value format") from e

            # Send the write command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_VWREG,
                addr=self.identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()

            # Check the response for errors
            if r.packet_type == OW_ERROR:
                logger.error("Error verifying writing TX register value")
                return False

            logger.info(f"Successfully wrote value 0x{value:08X} to register 0x{address:04X}")
            return True

        except ValueError as ve:
            logger.error(f"Validation error in write_register: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in write_register: {e}")
            raise

    def write_block_verify(self, start_address: int, reg_values: List[int]) -> bool:
        """
        Write a block of register values to the TX device with verification.

        Args:
            start_address (int): The starting register address to write to.
            reg_values (List[int]): List of register values to write.

        Returns:
            bool: True if the block write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, the identifier is invalid, or parameters are out of range.
        """
        try:
            # Ensure the UART connection is active
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if self.identifier < 0:
                raise ValueError("TX Chip address NOT SET")
            if self.identifier > 1:
                raise ValueError("TX Chip address must be in the range 0-1")

            # Validate the reg_values list
            if not reg_values or not isinstance(reg_values, list):
                raise ValueError("Invalid register values: Must be a non-empty list of integers")
            if any(not isinstance(value, int) for value in reg_values):
                raise ValueError("Invalid register values: All elements must be integers")

            # Configure chunking for large blocks
            max_regs_per_block = 62  # Maximum registers per block due to payload size
            num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
            logger.info(f"Write Block: Total chunks = {num_chunks}")

            # Write each chunk
            for i in range(num_chunks):
                chunk_start = i * max_regs_per_block
                chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
                chunk = reg_values[chunk_start:chunk_end]

                # Pack the chunk into the required data format
                try:
                    data_format = '<HBB' + 'I' * len(chunk)  # Start address (H), chunk length (B), reserved (B), values (I...)
                    data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)
                except struct.error as e:
                    logger.error(f"Error packing data for chunk {i}: {e}")
                    return False

                # Send the packet
                r = self.uart.send_packet(
                    id=None,
                    packetType=OW_TX7332,
                    command=OW_TX7332_VWBLOCK,
                    addr=self.identifier,
                    data=data
                )

                # Clear the UART buffer after sending
                self.uart.clear_buffer()

                # Check for errors in the response
                if r.packet_type == OW_ERROR:
                    logger.error(f"Error verifying writing TX block at chunk {i}")
                    return False

            logger.info("Block write successful")
            return True

        except ValueError as ve:
            logger.error(f"Validation error in write_block: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in write_block: {e}")
            raise

    def apply_ti_config_file(self, txchip_id:int, file_path:str) -> bool:
        """
        Reads a TI configuration file and writes the parsed registers to the device.

        :param file_path: Path to the TI config file.
        """


        # Validate the identifier
        if txchip_id < 0:
            raise ValueError("TX Chip address NOT SET")
        if txchip_id > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            parsed_registers = self.__parse_ti_cfg_file(file_path)

            for group, addr, value in parsed_registers:
                logger.info(f"{group:<20}0x{addr:02X}      0x{value:08X}")
                if not self.write_register(identifier=txchip_id, address=addr, value=value):
                    logger.error(f"Error writing TX CHIP ID: {txchip_id} register 0x{addr:02X} with value 0x{value:08X}")
                    return False

            return True

        except Exception as e:
            logger.error("Error parsing and writing TI config to TX Device: %s", e)
            raise

    @property
    def tx_devices(self) -> list[TX7332_IF]:
        """
        Get the list of enumerated TX devices.

        Returns:
            list[TX7332_IF]: The list of TX7332 interface instances.
        """
        return self._tx_instances

    @property
    def print(self) -> None:
        """
        Print TX device information.

        Raises:
            None
        """
        print("TX Device Information") # noqa: T201
        print("  UART Port:") # noqa: T201
        self.uart.print()
