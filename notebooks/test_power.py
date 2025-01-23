
import asyncio
import sys

from openlifu.io.core import UART
from openlifu.io.pwr_if import PWR_IF
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
        pid = 0x57A0  # Example PID for demonstration

        com_port = list_vcp_with_vid_pid(vid, pid)
        if com_port is None:
            print("No device found")
            sys.exit()
        else:
            print("Device found at port: ", com_port)
            # Select communication port
            s = UART(com_port, timeout=5)
    else:
        s = UART(PORT_NAME, timeout=5)

    pwr_if = PWR_IF(s)
    count = 0

    print("LIFU Power Controller Test")
    print("Ping Controller")
    # Send and Receive General ping command
    r = await pwr_if.ping()

    # Format and print the received data in hex format
    format_and_print_hex(r)

    print("Version Controller")
    # Send and Receive General ping command
    r = await pwr_if.version()

    # Format and print the received data in hex format
    format_and_print_hex(r)

    print("Echo Controller")
    # Send and Receive General ping command
    r = await pwr_if.echo(data=b'Hello World')

    # Format and print the received data in hex format
    format_and_print_hex(r)

    print("Toggle LED Controller")
    # Send and Receive General ping command
    r = await pwr_if.toggle_led()

    # Format and print the received data in hex format
    format_and_print_hex(r)

    print("CHIP ID Controller")
    # Send and Receive General ping command
    r = await pwr_if.chipid()

    # Format and print the received data in hex format
    format_and_print_hex(r)

    # Turn 12V Power ON
    print("Turn 12V Power ON")
    r = await pwr_if.set_12v_on()
    format_and_print_hex(r)

    input("Press [ENTER] key to Turn 12V Off...")

    # Turn 12V Power OFF
    print("Turn HV Power OFF")
    r = await pwr_if.set_12v_off()
    format_and_print_hex(r)
    
    # Set HV Power
    print("Set HV Power")
    r = await pwr_if.set_hv_supply(dac_input=1200)
    format_and_print_hex(r)

    # Turn HV Power ON
    print("Turn HV Power ON")
    r = await pwr_if.set_hv_supply_on()
    format_and_print_hex(r)

    input("Press [ENTER] key to Turn Power Off...")

    # Turn HV Power OFF
    print("Turn HV Power OFF")
    r = await pwr_if.set_hv_supply_off()
    format_and_print_hex(r)

    s.close()

asyncio.run(main())
