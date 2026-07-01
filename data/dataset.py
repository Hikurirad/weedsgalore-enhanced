"""WeedsGalore dataset with multi-channel input support.

Supports 4 input modes derived from 5-band UAV multispectral imagery
(R, G, B, NIR, RedEdge):

    rgb (3ch)   — R, G, B visible bands
    msi (5ch)   — R, G, B, NIR, RE  (official baseline input)
    vi  (5ch)   — R, G, B, NDVI, NDRE  (vegetation indices replace NIR/RE)
    msi_vi (7ch)— R, G, B, NIR, RE, NDVI, NDRE  (best-performing input)

Vegetation indices:
    NDVI = (NIR - R) / (NIR + R + eps)      [Rouse et al., 1974]
    NDRE = (NIR - RE) / (NIR + RE + eps)    [Gitelson & Merzlyak, 1994]

Images are read with matplotlib.pyplot.imread which normalizes 16-bit PNG
to float32 [0, 1].  Labels are read from semantics/ via PIL (integer class
indices, not divided by 255).
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class WeedsGaloreDataset(Dataset):
    """WeedsGalore UAV multispectral crop/weed segmentation dataset.

    Dataset: Celikkan et al., WeedsGalore (WACV 2025)
    https://github.com/GFZ/weedsgalore
    """

    _CH_MAP = {'rgb': 3, 'msi': 5, 'vi': 5, 'msi_vi': 7}

    def __init__(self, dataset_path, input_mode='msi', num_classes=3,
                 is_training=True, split='train', augmentation=True,
                 dataset_size=None, splits_dir=None):
        assert input_mode in self._CH_MAP, \
            f"input_mode must be one of {list(self._CH_MAP)}, got '{input_mode}'"
        assert num_classes in (3, 6), \
            "num_classes must be 3 (uni-weed) or 6 (multi-weed)"

        self.dataset_path = dataset_path
        self.input_mode = input_mode
        self.num_classes = num_classes
        self.augmentation = augmentation and is_training
        self.in_channels = self._CH_MAP[input_mode]

        if splits_dir is None:
            splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'splits')
        split_file = os.path.join(splits_dir, f'{split}.txt')
        with open(split_file) as f:
            names = [l.strip() for l in f if l.strip()]

        if dataset_size is not None:
            names = names[:dataset_size]
        self.names = names

    @property
    def in_bands(self):
        return self.in_channels

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        date = name[:10]
        img_dir = os.path.join(self.dataset_path, date, 'images')
        sem_dir = os.path.join(self.dataset_path, date, 'semantics')

        def _load(band):
            return plt.imread(os.path.join(img_dir, f'{name}_{band}.png'))

        red, green, blue = _load('R'), _load('G'), _load('B')
        nir, re_band = _load('NIR'), _load('RE')

        eps = 1e-6
        ndvi = (nir - red)    / (nir + red    + eps)
        ndre = (nir - re_band) / (nir + re_band + eps)

        if self.input_mode == 'rgb':
            channels = [red, green, blue]
        elif self.input_mode == 'msi':
            channels = [red, green, blue, nir, re_band]
        elif self.input_mode == 'vi':
            channels = [red, green, blue, ndvi, ndre]
        else:  # msi_vi
            channels = [red, green, blue, nir, re_band, ndvi, ndre]

        image = np.stack(channels, axis=0).astype(np.float32)  # (C, H, W)

        # Semantic label: integer class indices from semantics/
        label = np.array(Image.open(os.path.join(sem_dir, f'{name}.png')))

        # 3-class: bg=0, crop=1, weed=2 (all weed species merged)
        binary_label = np.where(label > 1, 2, label).astype(np.int64)
        # 6-class: bg=0, crop=1, weed_1..4=2..5
        unique_label = label.astype(np.int64)

        if self.augmentation:
            image, unique_label, binary_label = self._augment(
                image, unique_label, binary_label
            )

        return (torch.from_numpy(image),
                torch.from_numpy(unique_label),
                torch.from_numpy(binary_label))

    @staticmethod
    def _augment(image, unique_label, binary_label):
        """Data augmentation matching the official WeedsGalore protocol.

        Applies (in order): random rotation → random flip → Gaussian jitter.
        Matches augment_data() from the official repository:
        https://github.com/GFZ/weedsgalore/blob/main/src/utils.py
        """
        # Rotation: uniform choice of 0/90/180/270 degrees
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k=k, axes=(-2, -1)).copy()
            unique_label = np.rot90(unique_label, k=k, axes=(-2, -1)).copy()
            binary_label = np.rot90(binary_label, k=k, axes=(-2, -1)).copy()
        # Flip: mutually exclusive horizontal / vertical / none
        flip = np.random.choice([0, 1, 2])
        if flip == 1:   # vertical
            image = np.flip(image, axis=-2).copy()
            unique_label = np.flip(unique_label, axis=-2).copy()
            binary_label = np.flip(binary_label, axis=-2).copy()
        elif flip == 2:  # horizontal
            image = np.flip(image, axis=-1).copy()
            unique_label = np.flip(unique_label, axis=-1).copy()
            binary_label = np.flip(binary_label, axis=-1).copy()
        # Gaussian jitter: σ = 1/25, same as official implementation
        jitter = (np.random.randn(*image.shape) / 25).astype(np.float32)
        image = image + jitter
        return image, unique_label, binary_label
