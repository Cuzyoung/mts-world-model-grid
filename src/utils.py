"""Utility functions."""

import os
import yaml
import random
import numpy as np
import torch


def load_config(config_path, overrides=None):
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            config[k] = v
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
