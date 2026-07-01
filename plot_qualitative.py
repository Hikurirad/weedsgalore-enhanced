"""Generate qualitative prediction comparison: RGB / GT / A3 / M4."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import WeedsGaloreDataset
from models import deeplabv3plus_resnet50, SegModelWithInputAttention
from evaluate import replace_conv1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Color map: 0=bg (gray), 1=crop (green), 2=weed (red)
CMAP = np.array([[180, 180, 180], [60, 180, 60], [220, 50, 50]], dtype=np.uint8)

def load_model(ckpt, input_mode, use_attention=False, conv1_init='partial_random'):
    ch = {'rgb':3,'msi':5,'vi':5,'msi_vi':7}[input_mode]
    net = deeplabv3plus_resnet50(num_classes=3, pretrained_backbone=False)
    if ch != 3:
        net = replace_conv1(net, ch, DEVICE, init_mode=conv1_init)
    if use_attention:
        net = SegModelWithInputAttention(net, ch)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net = net.to(DEVICE).eval()
    return net

def predict(net, features):
    with torch.no_grad():
        out = net(features.unsqueeze(0).to(DEVICE))
    return torch.argmax(out, 1).squeeze(0).cpu().numpy()

def label2rgb(label):
    return CMAP[label.clip(0, 2)]

os.makedirs('outputs/figures', exist_ok=True)

# Load test dataset
ds_rgb   = WeedsGaloreDataset('../weedsgalore-dataset', input_mode='rgb',    num_classes=3, is_training=False, split='test', augmentation=False)
ds_msivi = WeedsGaloreDataset('../weedsgalore-dataset', input_mode='msi_vi', num_classes=3, is_training=False, split='test', augmentation=False)

# Load models
net_a3 = load_model('outputs/A3/best.pth',        'msi_vi')
net_m4 = load_model('outputs/M4-a3_ce_dice/best.pth', 'msi_vi')

# Pick samples: indices chosen to show diverse scene content
sample_indices = [3, 10, 18]
n = len(sample_indices)

fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
cols = ['RGB Image', 'Ground Truth (GT)', 'A3: MSI+VI + CE', 'M4: MSI+VI + CE-Dice']

for col_idx, title in enumerate(cols):
    axes[0, col_idx].set_title(title, fontsize=11, pad=6)

for row, idx in enumerate(sample_indices):
    feat_rgb,   _, bl_rgb   = ds_rgb[idx]
    feat_msivi, _, bl_msivi = ds_msivi[idx]

    rgb_img = feat_rgb[:3].permute(1, 2, 0).numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)

    gt = bl_rgb.numpy()
    pred_a3 = predict(net_a3, feat_msivi)
    pred_m4 = predict(net_m4, feat_msivi)

    axes[row, 0].imshow(rgb_img)
    axes[row, 1].imshow(label2rgb(gt))
    axes[row, 2].imshow(label2rgb(pred_a3))
    axes[row, 3].imshow(label2rgb(pred_m4))

    for col in range(4):
        axes[row, col].axis('off')

# Legend
legend_patches = [
    mpatches.Patch(color=CMAP[0]/255, label='Background'),
    mpatches.Patch(color=CMAP[1]/255, label='Crop'),
    mpatches.Patch(color=CMAP[2]/255, label='Weed'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('outputs/figures/qualitative.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/figures/qualitative.png')
