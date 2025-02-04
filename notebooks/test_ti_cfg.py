# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: ustx
#     language: python
#     name: python3
# ---
import asyncio
import time

import numpy as np

from openlifu.io.core import UART
from openlifu.io.ctrl_if import CTRL_IF
from openlifu.io.ustx import (
    DelayProfile,
    PulseProfile,
    TxModule,
    print_regs,
)
from openlifu.io.utils import list_vcp_with_vid_pid
from openlifu.xdc import Transducer


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

    # enumerate devices
    print("Enumerate TX Chips")
    r = await ustx_ctrl.enum_tx7332_devices()
    print("TX Device Count:", len(r))

    # load config 
    r = await ustx_ctrl.enum_tx7332_devices()
    for tx in ustx_ctrl.tx_devices:
        print(f"TX{tx.identifier} write registers")
        r = await tx.write_ti_config_file(file_path="notebooks/ti_example.cfg")
        

    print("Turn Trigger On")
    await ustx_ctrl.start_trigger()

    input("Press [ENTER] key to Stop...")

    print("Turn Trigger Off")
    await ustx_ctrl.stop_trigger()

    s.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
