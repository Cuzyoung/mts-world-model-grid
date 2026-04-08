"""
Training script for MTS-WM and baseline models.

Usage:
    python src/train.py --config configs/etth1.yaml
    python src/train.py --config configs/etth1.yaml --model dlinear
    python src/train.py --config configs/etth1.yaml --pred_len 336
"""

import argparse
import os
import sys
import json
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders
from src.metrics import compute_metrics
from src.utils import load_config, set_seed, get_device, count_parameters, ensure_dir
from src.models.world_model import (
    MTSWorldModel, SingleScaleWorldModel,
    NoSlowDynamicsModel, NoFastDynamicsModel,
)
from src.models.baselines import LSTMBaseline, DLinearBaseline, InformerBaseline


MODEL_REGISTRY = {
    "mts_wm": MTSWorldModel,
    "single_scale": SingleScaleWorldModel,
    "no_slow": NoSlowDynamicsModel,
    "no_fast": NoFastDynamicsModel,
    "lstm": LSTMBaseline,
    "dlinear": DLinearBaseline,
    "informer": InformerBaseline,
}


def build_model(config):
    model_name = config.get("model", "mts_wm")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name](config)
    return model


def train_epoch(model, loader, optimizer, device, kl_weight=0.01):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred, kl_loss = model(x)
        recon_loss = criterion(pred, y)
        loss = recon_loss + kl_weight * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += recon_loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = compute_metrics(preds, trues)
    return metrics, preds, trues


def main():
    parser = argparse.ArgumentParser(description="Train MTS-WM or baselines")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--pred_len", type=int, default=None, help="Override prediction length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    if args.pred_len:
        config["pred_len"] = args.pred_len

    set_seed(args.seed)
    device = get_device()

    model_name = config.get("model", "mts_wm")
    dataset = config["dataset"]
    pred_len = config["pred_len"]

    print(f"=== Training {model_name} on {dataset}, pred_len={pred_len} ===")

    # Data
    train_loader, val_loader, test_loader, scaler = get_data_loaders(config)
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Model
    model = build_model(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Training
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    kl_weight = config.get("kl_weight", 0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    # Checkpoint path
    ckpt_dir = os.path.join("checkpoints", f"{dataset}_{model_name}")
    ensure_dir(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"pred{pred_len}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, kl_weight)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} | "
              f"{elapsed:.1f}s")

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics, _, _ = evaluate(model, test_loader, device)
    print(f"\n=== Test Results: MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f} ===")

    # Save results
    result_dir = os.path.join("results", dataset)
    ensure_dir(result_dir)
    result_path = os.path.join(result_dir, f"{model_name}_pred{pred_len}.json")
    result = {
        "model": model_name,
        "dataset": dataset,
        "pred_len": pred_len,
        "seq_len": config["seq_len"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "params": count_parameters(model),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
