"""Evaluation metrics for time series forecasting."""

import numpy as np


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def compute_metrics(pred, true):
    return {
        "mse": float(mse(pred, true)),
        "mae": float(mae(pred, true)),
    }
