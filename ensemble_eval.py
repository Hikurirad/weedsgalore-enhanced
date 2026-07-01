"""Top-k val checkpoint ensemble evaluation."""
import os, csv, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import WeedsGaloreDataset
from models import deeplabv3plus_resnet50, deeplabv3plus_resnet50_do, SegModelWithInputAttention
from evaluate import replace_conv1


def build_model(in_channels, num_classes, use_do, use_attn, conv1_init, device):
    net = (deeplabv3plus_resnet50_do if use_do else deeplabv3plus_resnet50)(
        num_classes=num_classes, pretrained_backbone=False
    )
    if in_channels != 3:
        net = replace_conv1(net, in_channels, device, init_mode=conv1_init)
    if use_attn:
        net = SegModelWithInputAttention(net, in_channels)
    return net.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep_csv', required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--dataset_path', default='../weedsgalore-dataset')
    ap.add_argument('--input_mode', default='msi')
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--conv1_init', default='partial_random')
    ap.add_argument('--use_attention', action='store_true')
    ap.add_argument('--dlv3p_do', action='store_true')
    ap.add_argument('--splits_dir', default=None)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--ignore_index', type=int, default=-1)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows = []
    with open(args.sweep_csv) as f:
        for row in csv.DictReader(f):
            rows.append({'epoch': int(row['epoch']), 'val_miou': float(row['val_miou'])})
    topk = sorted(rows, key=lambda r: r['val_miou'], reverse=True)[:args.topk]
    topk_epochs = [r['epoch'] for r in topk]
    print(f"Top-{args.topk} val epochs: {topk_epochs}")
    print(f"Val mIoU: {[r['val_miou'] for r in topk]}")

    ds = WeedsGaloreDataset(args.dataset_path, input_mode=args.input_mode,
                            num_classes=args.num_classes, is_training=False,
                            split='test', augmentation=False, splits_dir=args.splits_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    in_channels = ds.in_channels

    metric = MulticlassJaccardIndex(num_classes=args.num_classes, average=None,
                                    ignore_index=args.ignore_index).to(device)
    all_probs = None
    all_labels = []

    for epoch in topk_epochs:
        ckpt = os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        net = build_model(in_channels, args.num_classes, args.dlv3p_do,
                          args.use_attention, args.conv1_init, device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()

        probs_list, labels_list = [], []
        with torch.no_grad():
            for features, unique_labels, binary_labels in loader:
                labels = binary_labels if args.num_classes == 3 else unique_labels
                out = net(features.to(device))
                probs_list.append(F.softmax(out, dim=1).cpu())
                labels_list.append(labels)

        epoch_probs = torch.cat(probs_list, dim=0)
        if all_probs is None:
            all_probs = epoch_probs
            all_labels = labels_list
        else:
            all_probs = all_probs + epoch_probs

    all_probs = all_probs / len(topk_epochs)
    pred = torch.argmax(all_probs, dim=1)
    for i, labels in enumerate(all_labels):
        metric.update(pred[i:i+1].to(device), labels.to(device))

    ious = metric.compute() * 100
    class_names = ['bg','crop','weed'] if args.num_classes == 3 else \
                  ['bg','crop','weed_1','weed_2','weed_3','weed_4']
    print(f"\n=== Top-{args.topk} Val Ensemble | Test Set ===")
    print(f"mIoU: {ious.mean().item():.2f}%")
    for name, iou in zip(class_names, ious):
        print(f"  IoU {name}: {iou.item():.2f}%")

if __name__ == '__main__':
    main()
