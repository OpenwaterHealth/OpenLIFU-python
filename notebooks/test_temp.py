from __future__ import annotations

import asyncio
import struct
import time

from openlifu.io.core import UART, UartPacket
from openlifu.io.ctrl_if import CTRL_IF
from openlifu.io.utils import format_and_print_hex, list_vcp_with_vid_pid


async def main():
    # Select communication port

    # s = UART('COM9', timeout=5)
    # s = UART('COM31', timeout=5)
     #s = UART('COM16', timeout=5)
    CTRL_BOARD = True  # change to false and specify PORT_NAME for Nucleo Board
    PORT_NAME = "COM16"
    s = None

    if CTRL_BOARD:
        vid = 0x483  # Example VID for demonstration
        pid = 0x57AF  # Example PID for demonstration

        com_port = list_vcp_with_vid_pid(vid, pid)
        if com_port is None:
            print("No device found")
        else:
            print("Device found at port: ", com_port)
            # Select communication port
            s = UART(com_port, timeout=5)
    else:
        s = UART(PORT_NAME, timeout=5)

    # Initialize the USTx controller object
    ustx_ctrl = CTRL_IF(s)

    print("Test PING")
    r = await ustx_ctrl.ping()
    format_and_print_hex(r)

    print("Get Temperature")
    for _ in range(10):  # Loop 10 times
        r = await ustx_ctrl.get_temperature()
        packet = UartPacket(buffer=r)
        if packet.data_len == 4:
            try:
                temperature = struct.unpack('<f', packet.data)[0]
                print("  Temperature (°C):", f"{temperature:.2f}")
            except Exception as e:
                print("  Error decoding float:", str(e))
        time.sleep(1)  # Sleep for 1 second

    s.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
