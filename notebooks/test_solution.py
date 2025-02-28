# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

pulse = Pulse(frequency=500e3, amplitude=1, duration=2e-5)
pt = Point(position=(0,0,30), units="mm")
sequence = Sequence(
    pulse_interval=0.1,
    pulse_count=10,
    pulse_train_interval=1,
    pulse_train_count=1
)
solution = Solution(
    id="solution",
    name="Solution",
    protocol_id="example_protocol",
    transducer_id="example_transducer",
    delays = np.zeros((1,64)),
    apodizations = np.ones((1,64)),
    pulse = pulse,
    sequence = sequence,
    target=pt,
    foci=[pt],
    approved=True
)


solution

ifx = LIFUInterface(test_mode=True)
ifx.txdevice.enum_tx7332_devices(_num_transmitters=2)

ifx.set_solution(solution)

txm = ifx.txdevice.tx_registers
r = {'DELAY CONTROL': {}, 'DELAY DATA': {}, 'PULSE CONTROL': {}, 'PULSE DATA': {}}
rtemplate = {}
profiles = ifx.txdevice.tx_registers.configured_pulse_profiles()
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
