"""Experiment configurations for WeedsGalore ablation study.

Final experiment set (7 experiments):

Input modality ablation (all CE, Top-5 val ensemble):
    A1  — RGB  (3ch)                    baseline
    A2  — MSI  (5ch, RGB+NIR+RE)        raw multispectral
    M1  — VI   (5ch, RGB+NDVI+NDRE)     vegetation indices only
    A3  — MSI+VI (7ch)                  multispectral + VI (best input)

Method ablation on MSI+VI (vs A3):
    M2  — MSI+VI + InputChannelAttention
    M3  — MSI+VI + partial_mean conv1 init
    M4  — MSI+VI + CE-Dice loss          ← best final method
"""

EXPERIMENTS = {
    # ---- Input modality ablation ----
    'A1': {
        'name': 'A1 — RGB Baseline',
        'description': 'Visible-light 3-channel baseline.',
        'input_mode': 'rgb',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce',
    },
    'A2': {
        'name': 'A2 — MSI (5ch)',
        'description': 'Raw multispectral: RGB + NIR + RedEdge.',
        'input_mode': 'msi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce',
    },
    'M1': {
        'name': 'M1 — RGB+VI (5ch)',
        'description': 'RGB + NDVI + NDRE; vegetation indices without raw NIR/RE.',
        'input_mode': 'vi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce',
    },
    'A3': {
        'name': 'A3 — MSI+VI (7ch)',
        'description': 'RGB + NIR + RE + NDVI + NDRE; best-performing input.',
        'input_mode': 'msi_vi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce',
    },

    # ---- Method ablation on MSI+VI ----
    'M2': {
        'name': 'M2 — MSI+VI + Attention',
        'description': 'Lightweight SE input channel attention on 7ch input.',
        'input_mode': 'msi_vi',
        'num_classes': 3,
        'use_attention': True,
        'conv1_init': 'partial_random',
        'loss_type': 'ce',
    },
    'M3': {
        'name': 'M3 — MSI+VI + partial_mean init',
        'description': 'RGB pretrained + mean-filled extra channels for conv1.',
        'input_mode': 'msi_vi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_mean',
        'loss_type': 'ce',
    },
    'M4': {
        'name': 'M4 — MSI+VI + CE-Dice (best)',
        'description': 'MSI+VI input with combined CE+Dice loss; best final method.',
        'input_mode': 'msi_vi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce_dice',
    },
    'M4-no-NDRE': {
        'name': 'M4-no-NDRE — MSI+NDVI without NDRE (6ch)',
        'description': 'Ablation: remove NDRE from M4 to validate channel sensitivity finding.',
        'input_mode': 'msi_ndvi',
        'num_classes': 3,
        'use_attention': False,
        'conv1_init': 'partial_random',
        'loss_type': 'ce_dice',
    },
}


def get_experiment_config(exp_name: str) -> dict:
    if exp_name not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment '{exp_name}'. Available: {available}"
        )
    return EXPERIMENTS[exp_name]
