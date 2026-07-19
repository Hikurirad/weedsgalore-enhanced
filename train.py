from absl import app, flags
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import WeedsGaloreDataset
from models import (
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet50_do,
    SegModelWithInputAttention,
)
from losses import get_loss
from utils.metrics import SegmentationMetrics

FLAGS = flags.FLAGS

# ---- Dataset ----
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('splits_dir', None, 'custom splits directory (None = default splits/)')
flags.DEFINE_integer('dataset_size_train', -1, 'limit train size; -1 = use all')

# ---- Input configuration ----
flags.DEFINE_string(
    'input_mode', 'msi',
    'Input mode: rgb (3ch), msi (5ch, RGB+NIR+RE), vi (5ch, RGB+NDVI+NDRE), '
    'msi_vi (7ch, RGB+NIR+RE+NDVI+NDRE), msi_ndvi (6ch, RGB+NIR+RE+NDVI, no NDRE)'
)
flags.DEFINE_boolean('use_attention', False,
                     'Wrap model with InputChannelAttention')

# ---- Model ----
flags.DEFINE_integer('num_classes', 3, '3 (uni-weed) or 6 (multi-weed)')
flags.DEFINE_boolean('dlv3p_do', False,
                     'Use Dropout variant of DeepLabV3+ (probabilistic)')
flags.DEFINE_boolean('pretrained_backbone', True,
                     'Use pretrained ResNet50 backbone')
flags.DEFINE_string('ckpt_resnet', 'ckpts/resnet50-19c8e357.pth',
                    'Path to pretrained backbone')
flags.DEFINE_string(
    'conv1_init', 'partial_random',
    'Conv1 init mode for non-RGB inputs: random (official), '
    'partial_random (RGB pretrained + extra random), partial_mean (RGB pretrained + extra mean-filled)'
)

# ---- Loss ----
flags.DEFINE_string('loss_type', 'ce',
                    'Loss: ce, focal, dice, ce_dice, focal_dice')
flags.DEFINE_list('class_weights', None,
                  'Per-class loss weights, e.g. 0.5,1.0,2.0')

# ---- Training ----
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_workers', 4, 'number of dataloader workers')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 10, 'number of training epochs')
flags.DEFINE_integer('start_epoch', 0, 'resume from this epoch index')
flags.DEFINE_string('resume_ckpt', '', 'resume model weights path')
flags.DEFINE_integer('ignore_index', -1, 'ignore label index')
flags.DEFINE_string('out_dir', 'out_dir', 'output directory for logs/ckpts')
flags.DEFINE_integer('log_interval', 25, 'iterations between log outputs')
flags.DEFINE_integer('ckpt_interval', 500, 'iterations between checkpoint saves')
flags.DEFINE_integer('seed', 42, 'random seed')


def replace_conv1(net, in_channels, device, init_mode="partial_random"):
    assert init_mode in ("random", "partial_random", "partial_mean"), \
        f"Unknown conv1_init: {init_mode}"
    old_conv = net.backbone.conv1

    if in_channels == old_conv.in_channels:
        return net

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
    )

    if init_mode == "random":
        net.backbone.conv1 = new_conv.to(device)
        return net

    # partial_random or partial_mean: keep RGB pretrained filters
    with torch.no_grad():
        copy_ch = min(3, old_conv.weight.shape[1], in_channels)
        new_conv.weight[:, :copy_ch, :, :] = old_conv.weight[:, :copy_ch, :, :]

        if in_channels > 3 and init_mode == "partial_mean":
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:, :, :] = mean_weight.repeat(1, in_channels - 3, 1, 1)
        # partial_random: extra channels keep default kaiming init

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    net.backbone.conv1 = new_conv.to(device)
    return net


def build_model(in_channels, num_classes, pretrained, use_do, use_attn,
                device, conv1_init="partial_random"):
    if use_do:
        net = deeplabv3plus_resnet50_do(
            num_classes=num_classes, pretrained_backbone=pretrained
        )
    else:
        net = deeplabv3plus_resnet50(
            num_classes=num_classes, pretrained_backbone=pretrained
        )

    if in_channels != 3:
        net = replace_conv1(net, in_channels, device, init_mode=conv1_init)

    if use_attn:
        net = SegModelWithInputAttention(net, in_channels)
    return net


def main(_):
    import random, numpy as np
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    input_mode = FLAGS.input_mode
    num_classes = FLAGS.num_classes

    # ---- Datasets ----
    train_size = FLAGS.dataset_size_train if FLAGS.dataset_size_train > 0 else None
    train_dataset = WeedsGaloreDataset(
        dataset_path=FLAGS.dataset_path,
        input_mode=input_mode,
        num_classes=num_classes,
        is_training=True,
        split='train',
        augmentation=True,
        dataset_size=train_size,
        splits_dir=FLAGS.splits_dir,
    )
    val_dataset = WeedsGaloreDataset(
        dataset_path=FLAGS.dataset_path,
        input_mode=input_mode,
        num_classes=num_classes,
        is_training=False,
        split='val',
        augmentation=False,
        splits_dir=FLAGS.splits_dir,
    )

    in_channels = train_dataset.in_channels
    print(f"Input mode: {input_mode}, channels: {in_channels}, classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=FLAGS.num_workers, drop_last=False,
    )

    # ---- Model ----
    net = build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=FLAGS.pretrained_backbone,
        use_do=FLAGS.dlv3p_do,
        use_attn=FLAGS.use_attention,
        device=device,
        conv1_init=FLAGS.conv1_init,
    )
    net = net.to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model params: {total_params / 1e6:.2f}M")

    # ---- Loss ----
    class_weights = None
    if FLAGS.class_weights:
        class_weights = [float(w) for w in FLAGS.class_weights]
        assert len(class_weights) == num_classes, \
            f"class_weights length {len(class_weights)} != num_classes {num_classes}"
    criterion = get_loss(
        loss_type=FLAGS.loss_type,
        num_classes=num_classes,
        class_weights=class_weights,
        ignore_index=FLAGS.ignore_index,
    ).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    if FLAGS.resume_ckpt:
        net.load_state_dict(torch.load(FLAGS.resume_ckpt, map_location=device))
        opt_path = os.path.join(os.path.dirname(FLAGS.resume_ckpt), 'optimizer.pth')
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"Resumed from {FLAGS.resume_ckpt}, starting at epoch {FLAGS.start_epoch}")

    # ---- Metrics ----
    train_metrics = SegmentationMetrics(
        num_classes=num_classes, ignore_index=FLAGS.ignore_index, device=device
    )
    val_metrics = SegmentationMetrics(
        num_classes=num_classes, ignore_index=FLAGS.ignore_index, device=device
    )

    # ---- Logging ----
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    writer = SummaryWriter(f'{FLAGS.out_dir}')
    print(f'Logging to: {FLAGS.out_dir}')

    best_miou = 0.0
    tot_iter = 0
    accum_loss = 0.0
    accum_iter = 0

    for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
        # ---- Train ----
        net.train()
        train_iter = iter(train_loader)
        for i, data in enumerate(train_iter):
            features, unique_labels, binary_labels = data
            labels = binary_labels if num_classes == 3 else unique_labels
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            out = net(features)
            loss = criterion(out, labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            _, pred = torch.max(out, 1)
            train_metrics.update(pred, labels)

            if tot_iter % FLAGS.log_interval == 0 or tot_iter == 1:
                metrics = train_metrics.compute()
                avg_loss = accum_loss / accum_iter
                print(f'Epoch: {epoch} iter: {tot_iter}, Loss: {avg_loss:.4f}, '
                      f'mIoU: {metrics["miou"]:.2f}%')

                writer.add_scalar('Train/Loss', avg_loss, tot_iter)
                writer.add_scalar('Train/mIoU', metrics["miou"], tot_iter)
                for idx, iou in enumerate(metrics["iou_per_class"]):
                    writer.add_scalar(f'Train/IoU_c{idx}', iou, tot_iter)

                train_metrics.reset()
                accum_loss, accum_iter = 0, 0

            if tot_iter % FLAGS.ckpt_interval == 0 or tot_iter == 1:
                torch.save(net.state_dict(), f'{FLAGS.out_dir}/ckpt_epoch{epoch}_iter{tot_iter}.pth')

        # Save epoch checkpoint
        torch.save(net.state_dict(), f'{FLAGS.out_dir}/epoch_{epoch}.pth')
        torch.save(optimizer.state_dict(), f'{FLAGS.out_dir}/optimizer.pth')

        # ---- Validation ----
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                features, unique_labels, binary_labels = data
                labels = binary_labels if num_classes == 3 else unique_labels
                features, labels = features.to(device), labels.to(device)
                out = net(features)
                _, pred = torch.max(out, 1)
                val_metrics.update(pred, labels)

        val_results = val_metrics.compute()
        val_miou = val_results["miou"]
        print(f"Epoch {epoch} Val mIoU: {val_miou:.2f}%")
        writer.add_scalar('Val/mIoU', val_miou, epoch)
        val_metrics.reset()

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(net.state_dict(), f'{FLAGS.out_dir}/best.pth')
            print(f"  New best model saved (mIoU={best_miou:.2f}%)")

    print(f"\nTraining complete. Best val mIoU: {best_miou:.2f}%")
    writer.close()

    # Save experiment config for reproducibility
    import json, subprocess
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'
    config = {
        'input_mode': FLAGS.input_mode,
        'num_classes': FLAGS.num_classes,
        'loss_type': FLAGS.loss_type,
        'conv1_init': FLAGS.conv1_init,
        'use_attention': FLAGS.use_attention,
        'dlv3p_do': FLAGS.dlv3p_do,
        'pretrained_backbone': FLAGS.pretrained_backbone,
        'epochs': FLAGS.epochs,
        'batch_size': FLAGS.batch_size,
        'lr': FLAGS.lr,
        'seed': FLAGS.seed,
        'dataset_path': FLAGS.dataset_path,
        'splits_dir': FLAGS.splits_dir,
        'dataset_size_train': FLAGS.dataset_size_train,
        'best_val_miou': round(best_miou, 4),
        'git_hash': git_hash,
    }
    with open(os.path.join(FLAGS.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {FLAGS.out_dir}/config.json")


if __name__ == '__main__':
    app.run(main)