# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyfus
#     language: python
#     name: python3
# ---

# Import the necessary modules
from ow_ustx import *
import json
import time
from pyfus.io.ustx import PulseProfile, DelayProfile, TxModule, TxArray, Tx7332Registers, print_regs
from pyfus.xdc import Transducer, Element
import numpy as np
import sys
from ow_ustx import *
import json
import time

# +
CTRL_BOARD = True  # change to false and specify PORT_NAME for Nucleo Board
PORT_NAME = "COM16"
serial_obj = None

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

# +
# Initialize the USTx controller object
ustx_ctrl = CTRL_IF(serial_obj)

print("USTx controller initialized")
r = await ustx_ctrl.ping()  # Ping the device
try:
    parsedResp = UartPacket(buffer=r)
    print("Received From Controller Packet ID: ", parsedResp.id)
except ValueError as e:
    print("{0}".format(e))
    sys.exit(0)
# -

# enumerate devices
print("Enumerate I2C Devices")
r = await ustx_ctrl.enum_i2c_devices()
print("I2C Device Count:", len(r))
print("Enumerate TX7332 Chips on AFE devices")
for afe_device in ustx_ctrl.afe_devices:
    print("Enumerate TX7332 Chips on AFE device")
    rUartPacket = await afe_device.enum_tx7332_devices()

# Set the focus and pulse profile
focus = np.array([0, 0, 50]) #set focus #left, front, down 
pulse_profile = PulseProfile(profile=1, frequency=400e3, cycles=2000)
first_afe = ustx_ctrl.afe_devices[0]
afe_dict = {first_afe.i2c_addr: first_afe}
# afe_dict = {afe.i2c_addr:afe for afe in ustx_ctrl.afe_devices}
# Load Mapping file
arr = Transducer.from_file(R"M3.json")
arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1))
tof = distances*1e-3 / 1500
delays = tof.max() - tof
i2c_addresses = list(afe_dict.keys())
i2c_addresses = i2c_addresses[:int(arr.numelements()/64)]
txa = TxArray(i2c_addresses=i2c_addresses)
array_delay_profile = DelayProfile(1, delays.tolist())
txa.add_delay_profile(array_delay_profile)
txa.add_pulse_profile(pulse_profile)
regs = txa.get_registers(profiles="configured", pack=True) 
for addr, rm in regs.items():
    print(f'I2C: 0x{addr:02x}')
    for i, r in enumerate(rm):
        print(f'MODULE {i}')
        print_regs(r)
    print('')  #calculate register state for 7332s, settings for board (bits, purpose), #change focus!!

# Write Registers to Device #series of loops for programming tx chips 
for i2c_addr, module_regs in regs.items():
    afe = afe_dict[i2c_addr] 
    for i in range(len(module_regs)):
        device = afe.tx_devices[i]
        r = module_regs[i]
        await device.write_register(0,1) #resetting the device
        for address, value in r.items():
            if isinstance(value, list):
                print(f"0x{i2c_addr:x}[{i}] Writing {len(value)}-value block starting at register 0x{address:X}")
                await device.write_block(address, value)
            else:
                print(f"0x{i2c_addr:x}[{i}] Writing value 0x{value:X} to register 0x{address:X}")
                await device.write_register(address, value)
            time.sleep(0.1)

# get trigger configuration
trigger_config = await ustx_ctrl.get_trigger()
print(json.dumps(trigger_config, indent=4))


# set trigger configuration
trigger_config = {
    "TriggerFrequencyHz": 50,
    "TriggerMode": 1,
    "TriggerPulseCount": 0,
    "TriggerPulseWidthUsec": 250
}
r = await ustx_ctrl.set_trigger(data=trigger_config)
format_and_print_hex(r)

# start trigger pulse
print("Turn Trigger On")
await ustx_ctrl.start_trigger()

# stop trigger pulse
print("Turn Trigger Off")
await ustx_ctrl.stop_trigger()

# close the communication port
ustx_ctrl.uart.close()
