from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlifu.io.LIFUInterface import LIFUInterface
    from openlifu.plan.solution import Solution

logger = logging.getLogger(__name__)

class LIFUDevice:
    def __init__(self, lifu_interface: LIFUInterface):
        """
        Initialize the LIFUDevice.

        Args:
            lifu_interface (LIFUInterface): The LIFUInterface instance for communication.
        """
        self.interface = lifu_interface

    def set_solution(self, solution: Solution):
        """
        Load a solution to the device.

        Args:
            solution (Solution): The solution to load.
        """
        try:
            logger.info("Loading solution: %s", solution.name)
            # Convert solution data and send to the device

            logger.info("Solution '%s' loaded successfully.", solution.name)
        except Exception as e:
            logger.error("Error loading solution '%s': %s", solution.name, e)
            raise

    def get_solution(self) -> Solution:
        """
        Retrieve the currently loaded solution from the device.

        Returns:
            Solution: The currently loaded solution.

        Raises:
            ValueError: If no solution is loaded.
        """
        try:
            logger.info("Retrieving the currently loaded solution.")
            # Example command to request solution metadata and data
            return None
        except Exception as e:
            logger.error("Error retrieving the solution: %s", e)
            raise


    def start_sonication(self):
        """
        Start sonication.

        Sets the device to a running state and sends a start command if necessary.
        """
        try:
            logger.info("Start Sonication")
            # Send the solution data to the device
        except Exception as e:
            logger.error("Error Starting sonication: %s", e)

    def stop_sonication(self):
        """
        Stop sonication.

        Stops the current sonication process.
        """
        try:
            logger.info("Stop Sonication")
            # Send the solution data to the device
        except Exception as e:
            logger.error("Error Stopping sonication: %s", e)
