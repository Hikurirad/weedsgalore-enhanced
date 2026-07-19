"""Sweep all epoch_*.pth checkpoints and report val/test mIoU per epoch."""
import os, re, csv, glob, argparse
import torch
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


@torch.no_grad()
def eval_split(net, dataset_path, input_mode, num_classes, split, device,
               ignore_index, splits_dir=None):
    ds = WeedsGaloreDataset(dataset_path=dataset_path, input_mode=input_mode,
                            num_classes=num_classes, is_training=False,
                            split=split, augmentation=False, splits_dir=splits_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    metric = MulticlassJaccardIndex(num_classes=num_classes, average=None,
                                    ignore_index=ignore_index).to(device)
    for features, unique_labels, binary_labels in loader:
        labels = binary_labels if num_classes == 3 else unique_labels
        out = net(features.to(device))
        metric.update(torch.argmax(out, 1), labels.to(device))
    ious = metric.compute() * 100
    return ious.mean().item(), [x.item() for x in ious]


def parse_epoch(p):
    m = re.search(r'epoch_(\d+)\.pth', os.path.basename(p))
    return int(m.group(1)) if m else 10**9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--dataset_path', default='../weedsgalore-dataset')
    ap.add_argument('--input_mode', default='msi')
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--conv1_init', default='partial_random')
    ap.add_argument('--use_attention', action='store_true')
    ap.add_argument('--dlv3p_do', action='store_true')
    ap.add_argument('--splits_dir', default=None)
    ap.add_argument('--out_csv', default='sweep.csv')
    ap.add_argument('--ignore_index', type=int, default=-1)
    ap.add_argument('--include_test', action='store_true',
                    help='Also compute test mIoU per epoch (for diagnostic analysis only; '
                         'do NOT use test results to select checkpoints)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'epoch_*.pth')), key=parse_epoch)
    if not ckpts:
        raise FileNotFoundError(f'No epoch_*.pth in {args.ckpt_dir}')

    ds_tmp = WeedsGaloreDataset(args.dataset_path, input_mode=args.input_mode,
                                num_classes=args.num_classes, is_training=False,
                                split='test', augmentation=False, splits_dir=args.splits_dir)
    in_channels = ds_tmp.in_channels

    rows = []
    for ckpt in ckpts:
        epoch = parse_epoch(ckpt)
        net = build_model(in_channels, args.num_classes, args.dlv3p_do,
                          args.use_attention, args.conv1_init, device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()

        val_miou, val_ious = eval_split(net, args.dataset_path, args.input_mode,
                                        args.num_classes, 'val', device,
                                        args.ignore_index, args.splits_dir)
        row = {'epoch': epoch, 'val_miou': val_miou}
        for i, v in enumerate(val_ious): row[f'val_c{i}'] = v

        if args.include_test:
            test_miou, test_ious = eval_split(net, args.dataset_path, args.input_mode,
                                              args.num_classes, 'test', device,
                                              args.ignore_index, args.splits_dir)
            row['test_miou'] = test_miou
            for i, v in enumerate(test_ious): row[f'test_c{i}'] = v
            print(f'ep{epoch:03d} | val {val_miou:.2f} | test {test_miou:.2f}')
        else:
            print(f'ep{epoch:03d} | val {val_miou:.2f}')

        rows.append(row)

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    best_val = max(rows, key=lambda r: r['val_miou'])
    print(f'\nBest val ep{best_val["epoch"]:03d}: val={best_val["val_miou"]:.2f}')
    if args.include_test:
        best_test = max(rows, key=lambda r: r['test_miou'])
        print(f'Best test ep{best_test["epoch"]:03d}: val={best_test["val_miou"]:.2f} '
              f'test={best_test["test_miou"]:.2f}')
        print('NOTE: test mIoU above is for diagnostic analysis only. '
              'Do NOT use test results to select checkpoints or K.')
    print(f'CSV: {args.out_csv}')

if __name__ == '__main__':
    main()
