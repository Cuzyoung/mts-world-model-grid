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
import logging

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders
from src.metrics import compute_metrics
from src.utils import load_config, set_seed, get_device, count_parameters, ensure_dir
from src.models.world_model import (
    MTSWorldModel, SingleScaleWorldModel,
    NoSlowDynamicsModel, NoFastDynamicsModel, NoDirect,
)
from src.models.baselines import LSTMBaseline, DLinearBaseline, InformerBaseline


MODEL_REGISTRY = {
    "mts_wm": MTSWorldModel,
    "single_scale": SingleScaleWorldModel,
    "no_slow": NoSlowDynamicsModel,
    "no_fast": NoFastDynamicsModel,
    "no_direct": NoDirect,
    "lstm": LSTMBaseline,
    "dlinear": DLinearBaseline,
    "informer": InformerBaseline,
}


def setup_logger(log_path):
    """Setup logger that writes to both file and console."""
    logger = logging.getLogger("mts_wm")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


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
    seq_len = config["seq_len"]

    # --- Setup directories ---
    exp_name = f"{dataset}_{model_name}_seq{seq_len}_pred{pred_len}"
    log_dir = os.path.join("logs", dataset, model_name)
    result_dir = os.path.join("results", dataset, model_name)
    ckpt_dir = os.path.join("checkpoints", dataset, model_name)
    ensure_dir(log_dir)
    ensure_dir(result_dir)
    ensure_dir(ckpt_dir)

    # --- Logger ---
    log_path = os.path.join(log_dir, f"pred{pred_len}.log")
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    logger.info(f"Device: {device}")

    # --- Data ---
    train_loader, val_loader, test_loader, scaler = get_data_loaders(config)
    logger.info(f"Data loaded: train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")

    # --- Model ---
    model = build_model(config).to(device)
    n_params = count_parameters(model)
    logger.info(f"Model: {model_name}, Parameters: {n_params:,}")

    # --- Training ---
    lr = config.get("lr", 1e-4)
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 10)
    kl_weight = config.get("kl_weight", 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = float("inf")
    best_val_mae = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(ckpt_dir, f"pred{pred_len}_best.pt")

    logger.info(f"Training: epochs={epochs}, lr={lr}, patience={patience}, kl_weight={kl_weight}")
    logger.info("-" * 60)

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, kl_weight)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_val_mae = val_metrics["mae"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  -> New best val_mse={best_val_mse:.6f}, checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - train_start
    logger.info("-" * 60)
    logger.info(f"Training finished in {train_time:.1f}s ({train_time/60:.1f}min)")
    logger.info(f"Best val_mse={best_val_mse:.6f}, val_mae={best_val_mae:.6f}")

    # --- Save val results ---
    val_result = {
        "model": model_name,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "train_time_sec": round(train_time, 1),
        "params": n_params,
    }
    val_result_path = os.path.join(result_dir, f"pred{pred_len}_val.json")
    with open(val_result_path, "w") as f:
        json.dump(val_result, f, indent=2)
    logger.info(f"Val results saved to {val_result_path}")

    # --- Test ---
    logger.info("=" * 60)
    logger.info("Testing on best checkpoint...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics, test_preds, test_trues = evaluate(model, test_loader, device)
    logger.info(f"TEST RESULTS: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}")

    # --- Save test results ---
    test_result = {
        "model": model_name,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "train_time_sec": round(train_time, 1),
        "params": n_params,
        "config": config,
    }
    test_result_path = os.path.join(result_dir, f"pred{pred_len}_test.json")
    with open(test_result_path, "w") as f:
        json.dump(test_result, f, indent=2)
    logger.info(f"Test results saved to {test_result_path}")

    # --- Save predictions (for later analysis) ---
    np.save(os.path.join(result_dir, f"pred{pred_len}_preds.npy"), test_preds)
    np.save(os.path.join(result_dir, f"pred{pred_len}_trues.npy"), test_trues)
    logger.info(f"Predictions saved: preds={test_preds.shape}, trues={test_trues.shape}")

    logger.info("=" * 60)
    logger.info("DONE")


if __name__ == "__main__":
    main()
