"""Generate Top-K ensemble curve plot for A1/A2/M1/A3/M4."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('outputs/figures', exist_ok=True)

K_vals = [1, 3, 5, 10]

data = {
    'A1 (RGB)':       [78.42, 79.78, 80.02, 79.79],
    'A2 (MSI)':       [77.36, 77.66, 76.75, 77.60],
    'M1 (RGB+VI)':    [79.90, 80.14, 81.10, 81.60],
    'A3 (MSI+VI)':    [80.56, 81.23, 82.38, 81.45],
    'M4 (MSI+VI\nCE-Dice)': [82.04, 83.80, 84.20, 83.57],
}

colors  = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
markers = ['o', 's', '^', 'D', '*']
msize   = [7, 7, 7, 7, 10]

fig, ax = plt.subplots(figsize=(7, 5))

for (label, vals), color, marker, ms in zip(data.items(), colors, markers, msize):
    ax.plot(K_vals, vals, marker=marker, color=color, linewidth=2,
            markersize=ms, label=label.replace('\n', ' '))

# Official B2 reference line
ax.axhline(y=82.90, color='gray', linestyle='--', linewidth=1.2,
           label='Official B2 (single ckpt ref.)')

ax.set_xlabel('K (number of checkpoints in ensemble)', fontsize=12)
ax.set_ylabel('Test mIoU (%)', fontsize=12)
ax.set_title('Top-K Validation Ensemble Performance on Test Set', fontsize=13)
ax.set_xticks(K_vals)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(74, 86)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9.5, loc='lower right')

# Annotate M4 peak at K=5
ax.annotate('84.20', xy=(5, 84.20), xytext=(5.4, 84.45),
            fontsize=9, color='#C4AD66',
            arrowprops=dict(arrowstyle='->', color='#C4AD66', lw=1.2))

plt.tight_layout()
plt.savefig('outputs/figures/topk_curve.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/figures/topk_curve.png')
