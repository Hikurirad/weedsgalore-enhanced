import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in weed segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        mask = targets != self.ignore_index
        return focal_loss[mask].mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for region-based optimization."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        mask = (targets != self.ignore_index).unsqueeze(1).float()
        targets_one_hot = targets_one_hot * mask
        probs = probs * mask

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """Combined segmentation loss.

    Supports CE, Focal, Dice, and weighted combinations.
    """

    def __init__(
        self,
        num_classes: int = 3,
        loss_type: str = 'ce',
        class_weights: list = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index

        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)

        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(ignore_index=ignore_index)
        elif loss_type == 'dice':
            self.criterion = DiceLoss(ignore_index=ignore_index)
        elif loss_type == 'ce_dice':
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            self.dice = DiceLoss(ignore_index=ignore_index)
        elif loss_type == 'focal_dice':
            self.focal = FocalLoss(ignore_index=ignore_index)
            self.dice = DiceLoss(ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'ce_dice':
            return self.ce(logits, targets) + self.dice(logits, targets)
        elif self.loss_type == 'focal_dice':
            return self.focal(logits, targets) + self.dice(logits, targets)
        return self.criterion(logits, targets)


def get_loss(
    loss_type: str = 'ce',
    num_classes: int = 3,
    class_weights: list = None,
    ignore_index: int = -1,
) -> SegLoss:
    return SegLoss(
        num_classes=num_classes,
        loss_type=loss_type,
        class_weights=class_weights,
        ignore_index=ignore_index,
    )