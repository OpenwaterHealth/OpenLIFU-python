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

# %%
tx = Tx7332Registers()
delays = np.arange(32)*1e-6
apodizations = np.ones(32)
delay_profile_1 = Tx7332DelayProfile(1, delays, apodizations)
tx.add_delay_profile(delay_profile_1)
print('CONTROL')
print_regs(tx.get_delay_control_registers())
print('DATA')
print_regs(tx.get_delay_data_registers())

# %%
frequency = 150e3
cycles = 3
pulse_profile_1 = Tx7332PulseProfile(1, frequency, cycles)
print('CONTROL')
tx.add_pulse_profile(pulse_profile_1)
print_regs(tx.get_pulse_control_registers())
print('DATA')
print_regs(tx.get_pulse_data_registers())

# %%
tx.add_delay_profile(delay_profile_1)
tx.add_pulse_profile(pulse_profile_1)
r = tx.get_registers(profiles="all")
print_regs(r)

# %%
x = np.linspace(-0.5, 0.5, 32)*4e-2
r = np.sqrt(x**2 + 5e-2**2)
delays = (r.max()-r)/1500
apodizations = [0,1]*16
delay_profile_2 = Tx7332DelayProfile(2, delays, apodizations)
tx.add_delay_profile(delay_profile_2)
print(f'{len(tx._delay_profiles_list)} Delay Profiles')
print(f'{len(tx._pulse_profiles_list)} Pulse Profiles')
print('')
for pack in [False, True]:
    print(f'Pack: {pack}')
    for profile_opts in ["active", "configured", "all"]:
        r = tx.get_registers(profiles=profile_opts, pack=pack)
        print(f'{profile_opts}: {len(r)} Writes')
    print('')

# %%
for index in [1,2]:
    tx.activate_delay_profile(index)
    print('\n')
    print(index)
    print('CONTROL')
    print_regs(tx.get_delay_control_registers())
    print('DATA')
    print_regs(tx.get_delay_data_registers())


# %%
txm = TxDeviceRegisters()
delays = np.arange(64)*1e-6
apodizations = np.ones(64)
module_delay_profile_1 = Tx7332DelayProfile(1, delays, apodizations)
txm.add_delay_profile(module_delay_profile_1)
txm.add_pulse_profile(pulse_profile_1)
for i, r in enumerate(txm.get_delay_control_registers()):
    print(f'CONTROL {i}')
    print_regs(r)
for i, r in enumerate(txm.get_delay_data_registers()):
    print(f'DATA {i}')
    print_regs(r)

# %%
module_delay_profile_1

# %%
x = np.linspace(-0.5, 0.5, 64)*4e-2
r = np.sqrt(x**2 + 5e-2**2)
delays = (r.max()-r)/1500
apodizations = [0,1]*32
module_delay_profile_2 = Tx7332DelayProfile(2, delays, apodizations)
frequency = 100e3
cycles = 200
pulse_profile_2 = Tx7332PulseProfile(2, frequency, cycles)
txm.add_delay_profile(module_delay_profile_2, activate=True)
txm.add_pulse_profile(pulse_profile_2, activate=True)
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

# %%
r = {'DELAY CONTROL': {}, 'DELAY DATA': {}, 'PULSE CONTROL': {}, 'PULSE DATA': {}}
rtemplate = {}
profiles = [1,2]
for index in profiles:
    rtemplate[index] = ['---------']*2
for index in profiles:
    txm.activate_delay_profile(index)
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
