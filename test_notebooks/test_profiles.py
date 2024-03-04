from pyfus.io.tx7332 import TX7332
import matplotlib.pyplot as plt
import numpy as np

tx = TX7332()
delays = np.arange(32)*1e-6
profile = 1
tx.set_delay_profile(delays=delays, units='s', profile=profile)
prof = tx.get_delay_profile(profile=profile)
print("[DELAY_CTRL]:")
for addr, val in prof['registers'].items():
    print(f'0x{addr:X}:x{val:08X}')
print("[DELAYS]")
for addr, val in prof['delay_registers'].items():
    print(f'0x{addr:X}:x{val:08X}')
