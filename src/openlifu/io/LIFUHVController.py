from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlifu.io.LIFUInterface import LIFUInterface

logger = logging.getLogger(__name__)

class HVController:
    def __init__(self, lifu_interface: LIFUInterface):
        """
        Initialize the HVController.

        Args:
            lifu_interface (LIFUInterface): The LIFUInterface instance for communication.
        """
        self.interface = lifu_interface

        # Initialize the high voltage state (should get this from device)
        self.output_voltage = 0.0
        self.is_hv_on = False

    def turn_on(self):
        """
        Turn on the high voltage.
        """
        try:
            if not self.interface.is_device_connected():
                raise ValueError("High voltage controller not connected")

            logger.info("Turning on high voltage.")

            self.is_hv_on = True
            logger.info("High voltage turned on successfully.")
        except Exception as e:
            logger.error("Error turning on high voltage: %s", e)
            raise

    def turn_off(self):
        """
        Turn off the high voltage.
        """
        try:
            if not self.interface.is_device_connected():
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
            if not self.interface.is_device_connected():
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
