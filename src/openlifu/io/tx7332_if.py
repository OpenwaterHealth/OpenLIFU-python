import logging
import struct
from typing import List

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


from openlifu.io.config import (
    OW_ERROR,
    OW_TX7332,
    OW_TX7332_RREG,
    OW_TX7332_VWBLOCK,
    OW_TX7332_VWREG,
    OW_TX7332_WBLOCK,
    OW_TX7332_WREG,
)


class TX7332_IF:

    def __init__(self, ctrl_if, identifier: int = -1):
        self.ctrl_if = ctrl_if
        self.uart = ctrl_if.uart
        self.identifier = identifier

    def get_index(self) -> int:
        return self.identifier

    def write_register(self, address: int, value: int) -> bool:
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
                command=OW_TX7332_WREG,
                addr=self.identifier,
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
            # Validate the connection
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

    def write_block(self, start_address: int, reg_values: List[int]) -> bool:
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
                    command=OW_TX7332_WBLOCK,
                    addr=self.identifier,
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

    def print(self):
        print("Controller Instance Information") # noqa: T201
        print(f"  Transmitter: {self.identifier}") # noqa: T201
