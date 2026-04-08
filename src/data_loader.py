"""
Data loaders for ETT and ECL datasets.
Follows the standard Informer train/val/test split protocol.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Generic sliding-window dataset for time series forecasting."""

    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_ett_data(root_path, dataset_name, seq_len, pred_len, batch_size=32):
    """
    Load ETT dataset (ETTh1, ETTh2, ETTm1).

    Standard split:
        ETTh1/ETTh2: train=8640, val=2880, test=2880
        ETTm1: train=34560, val=11520, test=11520
    """
    file_map = {
        "ETTh1": "ETTh1.csv",
        "ETTh2": "ETTh2.csv",
        "ETTm1": "ETTm1.csv",
    }
    filepath = os.path.join(root_path, "ETT-small", file_map[dataset_name])
    df = pd.read_csv(filepath)
    # Drop date column, keep numeric features
    data = df.iloc[:, 1:].values.astype(np.float32)

    # Standard split lengths
    if dataset_name in ["ETTh1", "ETTh2"]:
        train_end = 12 * 30 * 24          # 8640
        val_end = train_end + 4 * 30 * 24  # 8640 + 2880
    else:  # ETTm1
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data)

    train_data = data[:train_end]
    val_data = data[train_end - seq_len: val_end]  # overlap for context
    test_data = data[val_end - seq_len:]

    train_ds = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_ds = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scaler


def load_ecl_data(root_path, seq_len, pred_len, batch_size=32):
    """
    Load ECL (Electricity Consuming Load) dataset.
    321 clients, hourly data.
    Standard split: train 70%, val 10%, test 20%.
    """
    filepath = os.path.join(root_path, "ECL", "electricity.csv")
    df = pd.read_csv(filepath)
    data = df.iloc[:, 1:].values.astype(np.float32)

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data)

    train_data = data[:train_end]
    val_data = data[train_end - seq_len: val_end]
    test_data = data[val_end - seq_len:]

    train_ds = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_ds = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scaler


def get_data_loaders(config):
    """Dispatch to the correct data loader based on config."""
    dataset = config["dataset"]
    root_path = config.get("data_root", "data")
    seq_len = config["seq_len"]
    pred_len = config["pred_len"]
    batch_size = config.get("batch_size", 32)

    if dataset.startswith("ETT"):
        return load_ett_data(root_path, dataset, seq_len, pred_len, batch_size)
    elif dataset == "ECL":
        return load_ecl_data(root_path, seq_len, pred_len, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
