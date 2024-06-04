import sys
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'
    )
)
sys.path.append(parent_dir)

from txt_inputs.inputs import ARF, RMF, ENERGY_BINS

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fig, ax = plt.subplots(figsize=(12, 10))
# adding 0.00001 to avoid log(0) which is undefined
pcm = ax.imshow(
    RMF.T + 0.00001, 
    norm=LogNorm(vmin=RMF.min() + 0.00001, vmax=RMF.max()), 
    origin='lower', 
    aspect='auto', 
    cmap='gray'
)

fig.colorbar(pcm, ax=ax, label='Probability')
ax.set_title('RMF')

x_ticks = range(RMF.shape[0])[70::100] 

x_tick_labels = (ENERGY_BINS[70::100]).astype(int)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

ax.set_ylabel('Pulse-Height Channel')
ax.set_xlabel('Energy $E$')
plt.savefig('RMF.png')
plt.show()

plt.figure(figsize=(12,10))
plt.plot(ENERGY_BINS, ARF, color='black', lw=0.8)
plt.xlabel('Energy $E$')
plt.ylabel('Absorption Probability $A(E)$')
plt.savefig('ARF.png')