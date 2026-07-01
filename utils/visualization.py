import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    save_path: str = None,
    title: str = None,
):
    """Plot RGB image, ground truth, and prediction side by side.

    Args:
        image: (C, H, W) or (H, W, C) input image (first 3 channels shown as RGB).
        label: (H, W) ground truth label map.
        pred: (H, W) predicted label map.
        save_path: If provided, save figure to this path.
        title: Optional figure suptitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if image.shape[0] <= 8:
        rgb = image[:3].transpose(1, 2, 0)
    else:
        rgb = image
    rgb = np.clip(rgb, 0, 1)

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(label, cmap='jet', vmin=0, vmax=label.max())
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred, cmap='jet', vmin=0, vmax=pred.max())
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    confmat: np.ndarray,
    class_names: list = None,
    save_path: str = None,
):
    """Plot normalized confusion matrix.

    Args:
        confmat: (num_classes, num_classes) confusion matrix.
        class_names: List of class label strings.
        save_path: If provided, save figure to this path.
    """
    if class_names is None:
        class_names = [f'C{i}' for i in range(confmat.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confmat, cmap='Blues', vmin=0, vmax=1)

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, f'{confmat[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if confmat[i, j] > 0.5 else 'black')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)