import torch
import torch.nn as nn


class InputChannelAttention(nn.Module):
    """Lightweight SE-style channel attention for multi-spectral inputs.

    Learns per-channel importance weights via global average pooling
    followed by a bottleneck MLP. This lets the model adaptively
    weight different spectral bands and vegetation indices.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for bottleneck (default 4).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x))
        return x * weights


class SegModelWithInputAttention(nn.Module):
    """Wrapper that prepends input-channel attention to a segmentation model.

    The attention module re-weights input channels before they enter
    the DeepLabV3+ backbone, without modifying the original architecture.

    Args:
        base_model: The segmentation model (e.g. DeepLabV3+).
        in_channels: Number of input channels for the attention module.
        reduction: Reduction ratio for the attention bottleneck.
    """

    def __init__(self, base_model: nn.Module, in_channels: int, reduction: int = 4):
        super().__init__()
        self.attn = InputChannelAttention(in_channels, reduction=reduction)
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return self.base_model(x)