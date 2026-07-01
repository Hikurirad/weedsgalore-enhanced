import torch
import numpy as np
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)


class SegmentationMetrics:
    """Aggregated segmentation metrics for weed segmentation evaluation.

    Tracks: mIoU, per-class IoU, F1, Recall, and confusion matrix.
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = -1, device: str = 'cuda'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        self.iou = MulticlassJaccardIndex(
            num_classes=num_classes, average=None, ignore_index=ignore_index
        ).to(device)
        self.f1 = MulticlassF1Score(
            num_classes=num_classes, average=None, ignore_index=ignore_index
        ).to(device)
        self.recall = MulticlassRecall(
            num_classes=num_classes, average=None, ignore_index=ignore_index
        ).to(device)
        self.confmat = MulticlassConfusionMatrix(
            num_classes=num_classes, normalize='true', ignore_index=ignore_index
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.iou.update(preds, targets)
        self.f1.update(preds, targets)
        self.recall.update(preds, targets)
        self.confmat.update(preds, targets)

    def compute(self) -> dict:
        iou_scores = self.iou.compute()
        f1_scores = self.f1.compute()
        recall_scores = self.recall.compute()
        confmat = self.confmat.compute()
        return {
            'miou': iou_scores.mean().item() * 100,
            'iou_per_class': (iou_scores * 100).cpu().tolist(),
            'f1_per_class': (f1_scores * 100).cpu().tolist(),
            'recall_per_class': (recall_scores * 100).cpu().tolist(),
            'confusion_matrix': confmat.cpu().numpy(),
        }

    def compute_and_print(self, split: str = 'test', class_names: list = None):
        if class_names is None:
            class_names = ['bg', 'crop', 'weed'] if self.num_classes == 3 else [
                'bg', 'crop', 'weed_1', 'weed_2', 'weed_3', 'weed_4'
            ]

        results = self.compute()

        print(f"\n{'=' * 50}")
        print(f"{'Evaluation Results':^50}")
        print(f"{'=' * 50}")
        print(f"Split: {split}")
        print(f"mIoU: {results['miou']:.2f}%")
        for i, (name, iou) in enumerate(zip(class_names, results['iou_per_class'])):
            print(f"  IoU {name}: {iou:.2f}%")
        print(f"\nPer-class F1:")
        for i, (name, f1) in enumerate(zip(class_names, results['f1_per_class'])):
            print(f"  F1 {name}: {f1:.2f}%")
        print(f"\nPer-class Recall:")
        for i, (name, rec) in enumerate(zip(class_names, results['recall_per_class'])):
            print(f"  Recall {name}: {rec:.2f}%")
        print(f"\nConfusion Matrix:")
        print(np.array2string(
            results['confusion_matrix'], precision=3, suppress_small=True
        ))
        print(f"{'=' * 50}\n")

        return results

    def reset(self):
        self.iou.reset()
        self.f1.reset()
        self.recall.reset()
        self.confmat.reset()