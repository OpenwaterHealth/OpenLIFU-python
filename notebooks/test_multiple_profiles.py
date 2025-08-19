# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import numpy as np

from openlifu.io.LIFUTXDevice import (
    Tx7332DelayProfile,
    Tx7332PulseProfile,
    Tx7332Registers,
    TxDeviceRegisters,
    print_regs,
)

# Create delay profile
print("___TX registers___")
tx = Tx7332Registers()
delays = np.arange(32)*1e-6
apodizations = np.ones(32)

print("Creating a Delay Profile...\n")
delay_profile_1 = Tx7332DelayProfile(1, delays, apodizations)
tx.add_delay_profile(delay_profile_1)
print('DELAY CONTROL REGISTERS')
print_regs(tx.get_delay_control_registers())
print()
print('DELAY DATA REGISTERS')
print_regs(tx.get_delay_data_registers())
print()

# Create n number of pulse profiles
max_number_of_profiles = 16

number_of_profiles = 5

frequency = 150e3
cycles = 3

print("Creating Pulse Profiles...")
for i in range(1, number_of_profiles+1):
    pulse_profile = Tx7332PulseProfile(i + 1, frequency, cycles+10+i)
    tx.add_pulse_profile(pulse_profile)

print('PULSE CONTROL REGISTERS')
print_regs(tx.get_pulse_control_registers())
print('PULSE DATA REGISTERS')
print_regs(tx.get_pulse_data_registers())
print()

# %%
# print("Printing all registers...")
# r = tx.get_registers(profiles="all")
# print_regs(r)
# %%

x = np.linspace(-0.5, 0.5, 32)*4e-2
r = np.sqrt(x**2 + 5e-2**2)
delays = (r.max()-r)/1500
apodizations = [0,1]*16
print("Creating a second Delay Profile...")
delay_profile_2 = Tx7332DelayProfile(2, delays, apodizations)
tx.add_delay_profile(delay_profile_2)
print()
print(f'Delay Profiles: {len(tx._delay_profiles_list)} ')
print(f'Pulse Profiles: {len(tx._pulse_profiles_list)} ')
print('')
print("Getting registers with different packing options...")
for pack in [False, True]:
    print(f'Pack: {pack}')
    for profile_opts in ["active", "configured", "all"]:
        r = tx.get_registers(profiles=profile_opts, pack=pack)
        print(f'{profile_opts}: {len(r)} Writes')
    print('')

# %%
print("Activating Delay Profiles...")
print()
for index in [1,2]:
    tx.activate_delay_profile(index)
    print(f"Activating index {index}")
    print('DELAY CONTROL REGISTERS')
    print_regs(tx.get_delay_control_registers())
    print('DELAY DATA REGISTERS')
    print_regs(tx.get_delay_data_registers())
print()

print("___TX Device registers___")
# %%
txm = TxDeviceRegisters()
delays = np.arange(64)*1e-6
apodizations = np.ones(64)

print("Creating a Module Delay Profile...")
module_delay_profile_1 = Tx7332DelayProfile(1, delays, apodizations)
txm.add_delay_profile(module_delay_profile_1)

print(f"Adding {number_of_profiles} Pulse Profiles to Module...")
for i in range(1, number_of_profiles + 1):
    pulse_profile = Tx7332PulseProfile(i, frequency, cycles+10+i)
    txm.add_pulse_profile(pulse_profile)

print_data = False
if print_data:
    print('DELAY CONTROL REGISTERS')
    for i, r in enumerate(txm.get_delay_control_registers()):
        print(f'CONTROL {i}')
        print_regs(r)
    print('DELAY DATA REGISTERS')
    for i, r in enumerate(txm.get_delay_data_registers()):
        print(f'DATA {i}')
        print_regs(r)
else:
    print("Bypassing printing registers...")

# %%
print("Creating a second Module Delay Profile...")
x = np.linspace(-0.5, 0.5, 64)*4e-2
r = np.sqrt(x**2 + 5e-2**2)
delays = (r.max()-r)/1500
apodizations = [0,1]*32
module_delay_profile_2 = Tx7332DelayProfile(2, delays, apodizations)
frequency = 100e3
cycles = 200
# pulse_profile_2 = Tx7332PulseProfile(2, frequency, cycles)
# txm.add_delay_profile(module_delay_profile_2, activate=True)
# txm.add_pulse_profile(pulse_profile_2, activate=True)

if print_data:
    for i, r in enumerate(txm.get_delay_control_registers()):
        print(f'DELAY CONTROL {i}')
        print_regs(r)
    for i, r in enumerate(txm.get_delay_data_registers()):
        print(f'DELAY DATA {i}')
        print_regs(r)

    # %%
    for i, r in enumerate(txm.get_pulse_control_registers()):
        print(f'PULSE CONTROL {i}')
        print_regs(r)
    for i, r in enumerate(txm.get_pulse_data_registers()):
        print(f'PULSE DATA {i}')
        print_regs(r)
else:
    print("Bypassing printing registers...")

print()

# %%
r = {'DELAY CONTROL': {}, 'DELAY DATA': {}, 'PULSE CONTROL': {}, 'PULSE DATA': {}}
rtemplate = {}

profiles = list(range(1,number_of_profiles+1))
for index in profiles:
    rtemplate[index] = ['---------']*2
for index in profiles:
    txm.activate_delay_profile(1)
    txm.activate_pulse_profile(index)
    rdcm = txm.get_delay_control_registers()
    rddm = txm.get_delay_data_registers()
    rpcm = txm.get_pulse_control_registers()
    rpdm = txm.get_pulse_data_registers()
    d = {'DELAY CONTROL': rdcm, 'DELAY DATA': rddm, 'PULSE CONTROL': rpcm, 'PULSE DATA': rpdm}
    for k, rm in d.items():
        for txi, rc in enumerate(rm):
            for addr, value in rc.items():
                if addr not in r[k]:
                    r[k][addr] = {i: ['---------']*2 for i in profiles}
                r[k][addr][index][txi] = f'x{value:08x}'
h = [f'{"Profile " + str(i):19s}' for i in profiles]
print(f"      {' | '.join(h)}")
h1 = [f'{"TX" + str(i):>9s}' for i in [0,1]]
h1s = [' '.join(h1)]*len(profiles)
print(f"addr: {' | '.join(h1s)}")
for k, rm in r.items():
    print(f"{k}")
    for addr, rr in rm.items():
        print(f"x{addr:03x}: {' | '.join([' '.join(rr[i]) for i in profiles])}")

# %%
txm.get_registers(pack=False, pack_single=False)

for index in profiles:
    get_pulse_profile = txm.get_pulse_profile(index)
    print(f"Pulse Profile {index}: Frequency: {get_pulse_profile.frequency}, Cycles: {get_pulse_profile.cycles}")

txm.apply_all_registers()
