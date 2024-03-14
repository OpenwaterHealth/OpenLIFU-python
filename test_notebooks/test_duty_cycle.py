from pyfus.io.tx7332 import TX7332Registers
import matplotlib.pyplot as plt
import numpy as np

tx = TX7332Registers()
for duty_cycle in [0,.1,.25,.33,0.5,.66,.75,.9,1]:
    y,n,clk_div = tx.calc_pulse_pattern(100e3, duty_cycle)
    clk_freq = tx.bf_clk/(2**clk_div)
    t = np.arange(np.sum(np.array(n)+2))*(1/clk_freq)
    v = np.concatenate([[yi]*(ni+2) for yi,ni in zip(y,n)])
    plt.plot(t, v, '.-', label=f'{duty_cycle:.2f}, {np.sum(np.array(n)+2)}, {int(clk_freq/np.sum(np.array(n)+2)/1e3):.1f} kHz, {len(n)}, {clk_div}')
plt.legend()
print(f'{clk_freq/np.sum(np.array(n)+2)} Hz')
plt.show()
