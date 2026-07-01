"""Single-image inference for weed segmentation.

Usage:
    python inference.py --image_path=... --ckpt=... [--input_mode=msi] [--use_attention]
"""

from absl import app, flags
import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.wavelet import compute_wavelet_energy
from models import (
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet50_do,
    SegModelWithInputAttention,
)

FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', '', 'Path to input RGB image (H,W,3)')
flags.DEFINE_string('nir_path', None, 'Path to NIR band image')
flags.DEFINE_string('re_path', None, 'Path to RedEdge band image')
flags.DEFINE_string('input_mode', 'msi', 'rgb, msi, vi, msi_vi, msi_vi_wavelet')
flags.DEFINE_integer('num_classes', 3, '3 or 6')
flags.DEFINE_boolean('use_attention', False, 'Use attention wrapper')
flags.DEFINE_string('ckpt', '', 'Path to model checkpoint')
flags.DEFINE_string('output_path', 'prediction.png', 'Output segmentation path')
flags.DEFINE_integer('target_size', 640, 'Resize input to this square size')


COLORMAP = np.array([
    [0, 0, 0],       # 0: background
    [0, 255, 0],     # 1: crop (green)
    [255, 0, 0],     # 2: weed (red)
    [255, 255, 0],   # 3: weed_2
    [255, 0, 255],   # 4: weed_3
    [0, 255, 255],   # 5: weed_4
], dtype=np.uint8)


def build_input(image_rgb, nir, re_band, input_mode, target_size):
    """Build multi-channel input from individual bands."""
    h, w = target_size, target_size
    image_rgb = cv2.resize(image_rgb, (w, h)).astype(np.float32) / 255.0
    red, green, blue = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    eps = 1e-6

    channels = [red, green, blue]

    if input_mode in ('msi', 'msi_vi', 'msi_vi_wavelet'):
        if nir is None or re_band is None:
            raise ValueError(f"Input mode '{input_mode}' requires --nir_path and --re_path")
        nir = cv2.resize(nir, (w, h)).astype(np.float32) / 255.0
        re_band = cv2.resize(re_band, (w, h)).astype(np.float32) / 255.0
        channels.extend([nir, re_band])

    if input_mode in ('vi',):
        # For vi mode without NIR/RE, compute dummy NDVI/NDRE from NIR/RE args
        if nir is None or re_band is None:
            raise ValueError("vi mode requires --nir_path and --re_path")
        nir = cv2.resize(nir, (w, h)).astype(np.float32) / 255.0
        re_band = cv2.resize(re_band, (w, h)).astype(np.float32) / 255.0
        ndvi = (nir - red) / (nir + red + eps)
        ndre = (nir - re_band) / (nir + re_band + eps)
        channels = [red, green, blue, ndvi, ndre]
    elif input_mode in ('msi_vi', 'msi_vi_wavelet'):
        ndvi = (nir - red) / (nir + red + eps)
        ndre = (nir - re_band) / (nir + re_band + eps)
        channels.extend([ndvi, ndre])

    if input_mode == 'msi_vi_wavelet':
        wavelet_energy = compute_wavelet_energy(ndvi)
        channels.append(wavelet_energy)

    return np.stack(channels, axis=0).astype(np.float32)


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    in_channels_map = {
        'rgb': 3, 'msi': 5, 'vi': 5,
        'msi_vi': 7, 'msi_vi_wavelet': 8,
    }
    in_channels = in_channels_map[FLAGS.input_mode]

    net = deeplabv3plus_resnet50(
        num_classes=FLAGS.num_classes, pretrained_backbone=False
    )
    if in_channels != 3:
        net.backbone.conv1 = nn.Conv2d(
            in_channels, net.backbone.conv1.out_channels,
            kernel_size=7, stride=2, padding=3, bias=False, device=device,
        )
    if FLAGS.use_attention:
        net = SegModelWithInputAttention(net, in_channels)

    state = torch.load(FLAGS.ckpt, map_location=device)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    print("Model loaded.")

    # Load images
    image_rgb = cv2.imread(FLAGS.image_path)
    if image_rgb is None:
        raise FileNotFoundError(f"Image not found: {FLAGS.image_path}")
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    nir = None
    re_band = None
    if FLAGS.nir_path:
        nir = cv2.imread(FLAGS.nir_path, cv2.IMREAD_UNCHANGED)
        if nir is None:
            raise FileNotFoundError(f"NIR not found: {FLAGS.nir_path}")
    if FLAGS.re_path:
        re_band = cv2.imread(FLAGS.re_path, cv2.IMREAD_UNCHANGED)
        if re_band is None:
            raise FileNotFoundError(f"RE not found: {FLAGS.re_path}")

    # Build multi-channel input
    input_tensor = build_input(
        image_rgb, nir, re_band, FLAGS.input_mode, FLAGS.target_size
    )
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape} ({in_channels} channels)")

    # Inference
    with torch.no_grad():
        out = net(input_tensor)
        probs = torch.softmax(out, dim=1)
        _, pred = torch.max(out, 1)

    pred_np = pred[0].cpu().numpy().astype(np.uint8)

    # Colorize prediction
    colored = COLORMAP[pred_np]
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(FLAGS.output_path, colored)
    print(f"Prediction saved to: {FLAGS.output_path}")


if __name__ == '__main__':
    app.run(main)