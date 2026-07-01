"""Comprehensive evaluation script for weed segmentation.

Evaluates mIoU, per-class IoU, F1, Recall, and confusion matrix.
Supports all input modes and attention-wrapped models.
"""

from absl import app, flags
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import WeedsGaloreDataset
from models import (
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet50_do,
    SegModelWithInputAttention,
)
from utils.metrics import SegmentationMetrics
from utils.visualization import plot_prediction, plot_confusion_matrix

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('input_mode', 'msi',
                    'rgb, msi, vi, msi_vi, msi_vi_wavelet')
flags.DEFINE_integer('num_classes', 3, '3 (uni-weed) or 6 (multi-weed)')
flags.DEFINE_boolean('dlv3p_do', False, 'Dropout variant')
flags.DEFINE_boolean('use_attention', False, 'Attention wrapper')
flags.DEFINE_string('split', 'test', 'val or test')
flags.DEFINE_string('ckpt', '', 'Path to model checkpoint (.pth)')
flags.DEFINE_string('out_dir', 'eval_results', 'Output directory')
flags.DEFINE_integer('ignore_index', -1, 'Ignore label index')
flags.DEFINE_boolean('save_visualizations', False, 'Save prediction plots')
flags.DEFINE_string('splits_dir', None, 'custom splits directory (None = default splits/)')
flags.DEFINE_string(
    'conv1_init', 'partial_random',
    'Conv1 init mode for non-RGB inputs: random, partial_random, partial_mean'
)


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

    with torch.no_grad():
        copy_ch = min(3, old_conv.weight.shape[1], in_channels)
        new_conv.weight[:, :copy_ch, :, :] = old_conv.weight[:, :copy_ch, :, :]

        if in_channels > 3 and init_mode == "partial_mean":
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:, :, :] = mean_weight.repeat(1, in_channels - 3, 1, 1)

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    net.backbone.conv1 = new_conv.to(device)
    return net


def load_model(ckpt_path, in_channels, num_classes, use_do, use_attn, device,
               conv1_init="partial_random"):
    if use_do:
        net = deeplabv3plus_resnet50_do(
            num_classes=num_classes, pretrained_backbone=False
        )
    else:
        net = deeplabv3plus_resnet50(
            num_classes=num_classes, pretrained_backbone=False
        )

    if in_channels != 3:
        net = replace_conv1(net, in_channels, device, init_mode=conv1_init)

    if use_attn:
        net = SegModelWithInputAttention(net, in_channels)

    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    return net


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # Dataset
    dataset = WeedsGaloreDataset(
        dataset_path=FLAGS.dataset_path,
        input_mode=FLAGS.input_mode,
        num_classes=FLAGS.num_classes,
        is_training=False,
        split=FLAGS.split,
        augmentation=False,
        splits_dir=FLAGS.splits_dir,
    )
    in_channels = dataset.in_channels
    print(f"Input mode: {FLAGS.input_mode}, channels: {in_channels}")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=1, drop_last=False,
    )

    # Model
    net = load_model(
        FLAGS.ckpt, in_channels, FLAGS.num_classes,
        FLAGS.dlv3p_do, FLAGS.use_attention, device,
        conv1_init=FLAGS.conv1_init,
    )

    # Metrics
    metrics = SegmentationMetrics(
        num_classes=FLAGS.num_classes,
        ignore_index=FLAGS.ignore_index,
        device=device,
    )

    class_names = ['bg', 'crop', 'weed'] if FLAGS.num_classes == 3 else [
        'bg', 'crop', 'weed_1', 'weed_2', 'weed_3', 'weed_4'
    ]

    vis_dir = os.path.join(FLAGS.out_dir, 'visualizations')
    if FLAGS.save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)

    for i, data in enumerate(loader):
        features, unique_labels, binary_labels = data
        labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            out = net(features)

        probs = torch.softmax(out, dim=1)
        _, pred = torch.max(out, 1)
        metrics.update(pred, labels)

        if FLAGS.save_visualizations and i < 20:
            rgb = features[0, :3].cpu().numpy()
            label_np = labels[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            plot_prediction(
                rgb, label_np, pred_np,
                save_path=os.path.join(vis_dir, f'sample_{i}.png'),
                title=f'Sample {i}'
            )

    results = metrics.compute_and_print(split=FLAGS.split, class_names=class_names)

    confmat = results['confusion_matrix']
    plot_confusion_matrix(
        confmat, class_names,
        save_path=os.path.join(FLAGS.out_dir, 'confusion_matrix.png'),
    )
    print(f"Results saved to: {FLAGS.out_dir}")


if __name__ == '__main__':
    app.run(main)