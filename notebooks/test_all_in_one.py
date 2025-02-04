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

from openlifu.io.core import UART
from openlifu.io.ctrl_if import CTRL_IF
from openlifu.io.pwr_if import PWR_IF
from openlifu.io.utils import format_and_print_hex, list_vcp_with_vid_pid

async def main():
    # Select communication port

    TX_MODULE = True  # TX MODULE USB
    TX_PORT_NAME = "COM16"
    PWR_BOARD = True  # POWER CONSOLE USB
    PWR_PORT_NAME = "COM17"
    s_tx = None
    s_pwr = None

    if TX_MODULE:
        vid = 0x483  # Example VID for demonstration
        pid = 0x57AF  # Example PID for demonstration

        com_port = list_vcp_with_vid_pid(vid, pid)
        if com_port is None:
            print("No device found")
        else:
            print("Device found at port: ", com_port)
            # Select communication port
            s_tx = UART(com_port, timeout=5)
    else:
        s_tx = UART(TX_PORT_NAME, timeout=5)

    if PWR_BOARD:
        vid = 0x483  # Example VID for demonstration
        pid = 0x57A0  # Example PID for demonstration

        com_port = list_vcp_with_vid_pid(vid, pid)
        if com_port is None:
            print("No Power Console Device found")
        else:
            print("Power Console Device found at port: ", com_port)
            # Select communication port
            s_pwr = UART(com_port, timeout=5)
    else:
        s_pwr = UART(PWR_PORT_NAME, timeout=5)

    # Initialize the USTx controller object
    ustx_ctrl = CTRL_IF(s_tx)
    
    pwr_if = PWR_IF(s_pwr)

    print("LIFU Power Controller Test")
    print("Ping Controller")
    # Send and Receive General ping command
    r = await pwr_if.ping()

    # Turn 12V Power ON
    print("Turn 12V Power ON")
    r = await pwr_if.set_12v_on()
    
    
    # Set HV Power
    print("Set HV Power")
    r = await pwr_if.set_hv_supply(dac_input=1200)


    # enumerate devices
    print("Enumerate TX Chips")
    r = await ustx_ctrl.enum_tx7332_devices()
    print("TX Device Count:", len(r))

    # load config 
    r = await ustx_ctrl.enum_tx7332_devices()
    for tx in ustx_ctrl.tx_devices:
        print(f"TX{tx.identifier} write registers")
        r = await tx.write_ti_config_file(file_path="notebooks/ti_example.cfg")
        
    # Turn HV Power ON
    print("Turn HV Power ON")
    r = await pwr_if.set_hv_supply_on()
    format_and_print_hex(r)
    
    print("Turn Trigger On")
    await ustx_ctrl.start_trigger()

    input("Press [ENTER] key to Stop...")

    print("Turn Trigger Off")
    await ustx_ctrl.stop_trigger()

    # Turn HV Power OFF
    print("Turn HV Power OFF")
    r = await pwr_if.set_hv_supply_off()

    s_pwr.close()
    s_tx.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
