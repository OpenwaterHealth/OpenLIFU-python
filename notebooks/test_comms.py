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
    TxArray,
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


    focus = np.array([0, 0, 50]) #set focus #left, front, down
    pulse_profile = PulseProfile(profile=1, frequency=400e3, cycles=3)

    tx_dict = {tx.identifier: tx for tx in ustx_ctrl.tx_devices}

    arr = Transducer.from_file(R"E:\CURRENT-WORK\openwater\OpenLIFU-python\mappings\M3_rigidflex.json")
    arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
    distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1))
    tof = distances*1e-3 / 1500
    delays = tof.max() - tof
    tx_addresses = list(tx_dict.keys())
    tx_addresses = tx_addresses[:int(arr.numelements()/32)]
    txa = TxArray(i2c_addresses=tx_addresses)
    array_delay_profile = DelayProfile(1, delays.tolist())
    txa.add_delay_profile(array_delay_profile)
    txa.add_pulse_profile(pulse_profile)
    regs = txa.get_registers(profiles="configured", pack=True)
    for addr, rm in regs.items():
        print(f'TX_IDX: 0x{addr:02x}')
        for i, r in enumerate(rm):
            print(f'MODULE {i}')
            print_regs(r)
        print('')  #calculate register state for 7332s, settings for board (bits, purpose), #change focus!!

    # Write Registers to Device #series of loops for programming tx chips
    for tx_idx, module_regs in regs.items():
        tx_inst = tx_dict[tx_idx]
        r = module_regs[0]
        await tx_inst.write_register(0,1) #resetting the device
        for address, value in r.items():
            if isinstance(value, list):
                print(f"0x{tx_idx:x}[{i}] Writing {len(value)}-value block starting at register 0x{address:X}")
                await tx_inst.write_block(address, value)
            else:
                print(f"0x{tx_idx:x}[{i}] Writing value 0x{value:X} to register 0x{address:X}")
                await tx_inst.write_register(address, value)
            time.sleep(0.1)

    print("Turn Trigger On")
    await ustx_ctrl.start_trigger()

    input("Press [ENTER] key to Stop...")

    print("Turn Trigger Off")
    await ustx_ctrl.stop_trigger()

    s.close()

    #Visualize Delays
    from matplotlib import colormaps
    cm = colormaps.get("viridis")
    delays_norm = delays / delays.max()
    colors = cm(delays_norm)[:,:3]
    arr.draw(facecolor=colors)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
