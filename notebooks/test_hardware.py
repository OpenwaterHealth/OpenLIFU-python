# Import the necessary modules
from openlifu.io import UART
import json
import time
import numpy as np
import sys
import serial.tools.list_ports

CTRL_BOARD = True  # change to false and specify PORT_NAME for Nucleo Board
PORT_NAME = "COM16"
serial_obj = None

def list_vcp_with_vid_pid(target_vid, target_pid):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == target_vid and port.pid == target_pid:
            print(f"Device found: {port.device} - {port.description}")
            return port.device
    print("Device not found")
    return None

# Find the COM port of the device
if CTRL_BOARD:
    vid = 1155  # Example VID for demonstration
    pid = 22446  # Example PID for demonstration

    com_port = list_vcp_with_vid_pid(vid, pid)
    if com_port is None:
        print("No device found")
    else:
        print("Device found at port: ", com_port)
        # Select communication port
        serial_obj = UART(com_port, timeout=5)
else:
    serial_obj = UART(PORT_NAME, timeout=5)
