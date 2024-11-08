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

from pyfus.io.ustx import PulseProfile, DelayProfile, TxModule, Tx7332Registers, print_regs, TxArray, swap_byte_order
from pyfus.xdc import Transducer
import numpy as np
import sys
from ow_ustx import *
import json
import time

# Select communication port (Look up in Device Manager)
s = UART('COM17', timeout=5)
# s = UART('COM31', timeout=5)
# s = UART('COM34', timeout=5)
# create controller instance
ustx_ctrl = CTRL_IF(s)

# enumerate devices
print("Enumerate I2C Devices")
ustx_ctrl.enum_i2c_devices()
print("Enumerate TX7332 Chips on AFE devices")
for afe_device in ustx_ctrl.afe_devices:
    print("Enumerate TX7332 Chips on AFE device")
    rUartPacket = afe_device.enum_tx7332_devices()

focus = np.array([0, 0, 50]) #set focus #left, front, down 
pulse_profile = PulseProfile(profile=1, frequency=400e3, cycles=3)
afe_dict = {afe.i2c_addr:afe for afe in ustx_ctrl.afe_devices}
arr = Transducer.from_file(R"C:\Users\Neuromod2\Documents\20240416_USTXv2_v_VSX\M2.json")
arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1))
tof = distances*1e-3 / 1500
delays = tof.max() - tof 
txa = TxArray(i2c_addresses=list(afe_dict.keys()))
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
        device.write_register(0,1) #resetting the device
        for address, value in r.items():
            if isinstance(value, list):
                print(f"0x{i2c_addr:x}[{i}] Writing {len(value)}-value block starting at register 0x{address:X}")
                device.write_block(address, value)
            else:
                print(f"0x{i2c_addr:x}[{i}] Writing value 0x{value:X} to register 0x{address:X}")
                device.write_register(address, value)
            time.sleep(0.1)

print("Turn Trigger On")
ustx_ctrl.start_trigger()

print("Turn Trigger Off")
ustx_ctrl.stop_trigger()

s.close()

#Visualize Delays
from matplotlib import colormaps
cm = colormaps.get("viridis")
delays_norm = delays / delays.max()
colors = cm(delays_norm)[:,:3]
arr.draw(facecolor=colors)
