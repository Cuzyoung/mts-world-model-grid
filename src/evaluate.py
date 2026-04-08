"""
Evaluation script: load a trained checkpoint and evaluate on test set.

Usage:
    python src/evaluate.py --config configs/etth1.yaml --checkpoint checkpoints/ETTh1_mts_wm/pred96_best.pt
"""

import argparse
import os
import sys
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders
from src.metrics import compute_metrics
from src.utils import load_config, set_seed, get_device
from src.train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    if args.pred_len:
        config["pred_len"] = args.pred_len

    set_seed(42)
    device = get_device()

    _, _, test_loader, scaler = get_data_loaders(config)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = compute_metrics(preds, trues)

    model_name = config.get("model", "mts_wm")
    dataset = config["dataset"]
    pred_len = config["pred_len"]

    print(f"=== {model_name} on {dataset} (pred_len={pred_len}) ===")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
