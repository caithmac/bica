"""Quick plot: RF vs MLP parameter-matched comparison."""
import matplotlib.pyplot as plt
import numpy as np

rf_params = 6_141_755
rf_rmse = 1.0065

mlp_broken = [
    (66_945, 1.5027), (133_889, 1.5069), (267_777, 1.4996),
    (300_545, 1.5063), (666_625, 1.5136), (1_595_393, 1.4962),
    (1_726_465, 1.5135), (4_763_649, 1.5075), (8_435_713, 1.5084),
]
mlp_esm2 = [(333_121, 1.1389)]
mlp_shallow = [(268_546, 1.3285)]

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(*zip(*mlp_broken), c='#e74c3c', s=60, zorder=3, label='MLP + ECFP4 + AAC')
ax.scatter(*mlp_shallow[0], c='#e67e22', s=100, zorder=4, marker='s', label='MLP shallow + ECFP4 + AAC')
ax.scatter(*mlp_esm2[0], c='#3498db', s=100, zorder=4, marker='D', label='MLP + ECFP4 + ESM-2')
ax.axhline(y=rf_rmse, color='#2ecc71', linewidth=2.5, linestyle='--', zorder=2, label=f'RF + ECFP4 + AAC ({rf_rmse:.4f})')

ax.set_xscale('log')
ax.set_xlabel('Parameters')
ax.set_ylabel('Test RMSE')
ax.set_title('RF vs MLP: BindingDB scaffold split (seed 42)')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(5e4, 2e7)
ax.set_ylim(0.98, 1.55)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:/BICA/figures/rf_vs_mlp_capacity.png', dpi=150, bbox_inches='tight')
plt.savefig('E:/BICA/figures/rf_vs_mlp_capacity.svg', bbox_inches='tight')
