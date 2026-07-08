"""Channel Sensitivity Analysis: measure input channel sensitivity via mean replacement.

For each channel, replaces it with the channel mean (mean replacement, not zero) at inference time
and measures the drop in mIoU and weed IoU versus the baseline.

Usage:
    python channel_sensitivity.py \
        --ckpt=outputs/M4-a3_ce_dice/best.pth \
        --input_mode=msi_vi \
        --out_dir=outputs/channel_sensitivity
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import WeedsGaloreDataset
from models import deeplabv3plus_resnet50, SegModelWithInputAttention
from evaluate import replace_conv1
from torchmetrics.classification import MulticlassJaccardIndex

CHANNEL_NAMES = {
    'rgb':    ['R', 'G', 'B'],
    'msi':    ['R', 'G', 'B', 'NIR', 'RE'],
    'vi':     ['R', 'G', 'B', 'NDVI', 'NDRE'],
    'msi_vi': ['R', 'G', 'B', 'NIR', 'RE', 'NDVI', 'NDRE'],
}


def build_model(ckpt, input_mode, use_attention, conv1_init, device):
    ch = len(CHANNEL_NAMES[input_mode])
    net = deeplabv3plus_resnet50(num_classes=3, pretrained_backbone=False)
    if ch != 3:
        net = replace_conv1(net, ch, device, init_mode=conv1_init)
    if use_attention:
        net = SegModelWithInputAttention(net, ch)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def evaluate(net, ds, device, zero_channel=None, channel_mean=None):
    """Run inference and compute mIoU. Optionally zero out one channel."""
    metric = MulticlassJaccardIndex(num_classes=3, average=None,
                                    ignore_index=-1).to(device)
    for feat, _, bl in ds:
        x = feat.unsqueeze(0).to(device)
        if zero_channel is not None:
            x = x.clone()
            x[:, zero_channel, :, :] = channel_mean[zero_channel]
        out = net(x)
        pred = torch.argmax(out, 1)
        metric.update(pred, bl.unsqueeze(0).to(device))
    ious = metric.compute() * 100
    return {
        'miou': ious.mean().item(),
        'iou_bg': ious[0].item(),
        'iou_crop': ious[1].item(),
        'iou_weed': ious[2].item(),
    }


def compute_channel_means(ds):
    """Compute per-channel mean over the dataset for zero-replacement."""
    all_feats = torch.stack([ds[i][0] for i in range(len(ds))])  # (N,C,H,W)
    return all_feats.mean(dim=(0, 2, 3))  # (C,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input_mode', default='msi_vi')
    ap.add_argument('--conv1_init', default='partial_random')
    ap.add_argument('--use_attention', action='store_true')
    ap.add_argument('--dataset_path', default='../weedsgalore-dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out_dir', default='outputs/channel_sensitivity')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    net = build_model(args.ckpt, args.input_mode, args.use_attention,
                      args.conv1_init, device)
    ds = WeedsGaloreDataset(args.dataset_path, input_mode=args.input_mode,
                            num_classes=3, is_training=False,
                            split=args.split, augmentation=False)

    ch_names = CHANNEL_NAMES[args.input_mode]
    n_ch = len(ch_names)

    print('Computing channel means...')
    ch_mean = compute_channel_means(ds).to(device)

    print('Baseline evaluation (all channels intact)...')
    baseline = evaluate(net, ds, device)
    print(f"  Baseline mIoU={baseline['miou']:.2f}%  "
          f"weed IoU={baseline['iou_weed']:.2f}%")

    results = []
    for ch_idx, ch_name in enumerate(ch_names):
        r = evaluate(net, ds, device, zero_channel=ch_idx, channel_mean=ch_mean)
        drop_miou  = baseline['miou']  - r['miou']
        drop_weed  = baseline['iou_weed'] - r['iou_weed']
        drop_crop  = baseline['iou_crop'] - r['iou_crop']
        results.append({
            'channel': ch_name,
            'drop_miou': drop_miou,
            'drop_weed': drop_weed,
            'drop_crop': drop_crop,
            'miou': r['miou'],
        })
        print(f"  Zero {ch_name:6s}: ΔmIoU={drop_miou:+.2f}%  "
              f"Δweed={drop_weed:+.2f}%  Δcrop={drop_crop:+.2f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_ch)
    w = 0.55

    # mIoU drop
    drops_miou = [r['drop_miou'] for r in results]
    drops_weed = [r['drop_weed'] for r in results]
    colors_miou = ['#d65f5f' if d > 0 else '#6acc65' for d in drops_miou]
    colors_weed = ['#d65f5f' if d > 0 else '#6acc65' for d in drops_weed]

    axes[0].bar(x, drops_miou, width=w, color=colors_miou, edgecolor='white')
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ch_names, fontsize=11)
    axes[0].set_ylabel('mIoU drop (%) when channel replaced with mean', fontsize=10)
    axes[0].set_title(f'Channel Importance — mIoU\n'
                      f'(baseline {baseline["miou"]:.2f}%)', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, drops_weed, width=w, color=colors_weed, edgecolor='white')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ch_names, fontsize=11)
    axes[1].set_ylabel('weed IoU drop (%) when channel replaced with mean', fontsize=10)
    axes[1].set_title(f'Channel Importance — weed IoU\n'
                      f'(baseline {baseline["iou_weed"]:.2f}%)', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle(f'Channel Sensitivity Analysis — Mean Replacement  |  {args.input_mode.upper()} input  |  M4 (CE-Dice)',
                 fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f'channel_sensitivity_{args.input_mode}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out_path}')

    # Summary
    top_miou = sorted(results, key=lambda r: r['drop_miou'], reverse=True)
    top_weed = sorted(results, key=lambda r: r['drop_weed'], reverse=True)
    print(f'\nMost important channels (by mIoU drop):')
    for r in top_miou[:3]:
        print(f"  {r['channel']}: ΔmIoU={r['drop_miou']:+.2f}%")
    print(f'Most important channels (by weed IoU drop):')
    for r in top_weed[:3]:
        print(f"  {r['channel']}: Δweed={r['drop_weed']:+.2f}%")


if __name__ == '__main__':
    main()
