#!/usr/bin/env python
"""Experiment launcher — runs train + eval for a given experiment.

Usage:
    python experiments/run_experiment.py --exp=A5 --dataset_path=../weedsgalore-dataset
    python experiments/run_experiment.py --exp=B1,B2,B3 --dataset_path=... --epochs=20

    Or run all experiments:
    python experiments/run_experiment.py --exp=ALL --dataset_path=...

Experiment names:
    Group 1 (Baseline):   B1, B2, B3
    Group 2 (VI):         E1, E2
    Group 3 (Wavelet):    W1, W2
    Group 4 (Attention):  A1, A2, A3, A4, A5
"""

from absl import app, flags
import subprocess
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiments import EXPERIMENTS, get_experiment_config

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', 'A5', 'Experiment name(s), comma-separated, or "ALL"')
flags.DEFINE_string('dataset_path', '../weedsgalore-dataset',
                    'Path to dataset')
flags.DEFINE_integer('epochs', 10, 'Training epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_boolean('skip_train', False, 'Skip training, only evaluate')
flags.DEFINE_boolean('skip_eval', False, 'Skip evaluation, only train')


def run_experiment(exp_name: str, config: dict, epochs: int,
                   batch_size: int, lr: float, dataset_path: str):
    print(f"\n{'#' * 60}")
    print(f"# Running experiment: {exp_name} — {config.get('name', '')}")
    print(f"# Description: {config.get('description', '')}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(script_dir, 'train.py')
    eval_script = os.path.join(script_dir, 'evaluate.py')
    out_dir = os.path.join(script_dir, 'outputs', exp_name)

    train_flags = [
        f'--dataset_path={dataset_path}',
        f'--input_mode={config["input_mode"]}',
        f'--num_classes={config["num_classes"]}',
        f'--use_attention={config["use_attention"]}',
        f'--dlv3p_do={config.get("dlv3p_do", False)}',
        f'--loss_type={config.get("loss_type", "ce")}',
        f'--conv1_init={config.get("conv1_init", "partial_random")}',
        f'--epochs={epochs}',
        f'--batch_size={batch_size}',
        f'--lr={lr}',
        f'--out_dir={out_dir}',
    ]

    train_cmd = [sys.executable, train_script] + train_flags

    if not FLAGS.skip_train:
        print(f"Training command:\n{' '.join(train_cmd)}\n")
        result = subprocess.run(train_cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"ERROR: Training failed for {exp_name}")
            return

    # Eval — use best checkpoint if available, otherwise last epoch
    if not FLAGS.skip_eval:
        best_ckpt = os.path.join(out_dir, 'best.pth')
        last_ckpt = os.path.join(out_dir, f'epoch_{epochs - 1}.pth')
        ckpt = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
        if not os.path.exists(ckpt):
            print(f"WARNING: No checkpoint found at {ckpt}, skipping eval")
            return

        print(f"Evaluating with checkpoint: {ckpt}")

        eval_flags = [
            f'--dataset_path={dataset_path}',
            f'--input_mode={config["input_mode"]}',
            f'--num_classes={config["num_classes"]}',
            f'--use_attention={config["use_attention"]}',
            f'--dlv3p_do={config.get("dlv3p_do", False)}',
            f'--conv1_init={config.get("conv1_init", "partial_random")}',
            f'--split=test',
            f'--ckpt={ckpt}',
            f'--out_dir={out_dir}/eval_test',
            f'--save_visualizations=true',
        ]
        eval_cmd = [sys.executable, eval_script] + eval_flags
        result = subprocess.run(eval_cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"ERROR: Evaluation failed for {exp_name}")
            return

        # Also eval on val split
        eval_flags_val = [
            f'--dataset_path={dataset_path}',
            f'--input_mode={config["input_mode"]}',
            f'--num_classes={config["num_classes"]}',
            f'--use_attention={config["use_attention"]}',
            f'--dlv3p_do={config.get("dlv3p_do", False)}',
            f'--conv1_init={config.get("conv1_init", "partial_random")}',
            f'--split=val',
            f'--ckpt={ckpt}',
            f'--out_dir={out_dir}/eval_val',
        ]
        eval_cmd_val = [sys.executable, eval_script] + eval_flags_val
        subprocess.run(eval_cmd_val, cwd=script_dir)

    print(f"\nExperiment {exp_name} complete.\n")


def main(_):
    dataset_path = os.path.abspath(FLAGS.dataset_path)

    if FLAGS.exp == 'ALL':
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e.strip() for e in FLAGS.exp.split(',')]

    for exp_name in exp_names:
        config = get_experiment_config(exp_name)
        run_experiment(
            exp_name=exp_name,
            config=config,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            lr=FLAGS.lr,
            dataset_path=dataset_path,
        )

    print("\nAll experiments completed.")


if __name__ == '__main__':
    app.run(main)