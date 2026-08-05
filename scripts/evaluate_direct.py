from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssecast.direct import (
    CascadiaDirectDataset,
    ensure_direct_params,
    high_release_mask,
    load_checkpoint,
    load_params,
    make_model,
    weighted_nrmse_np,
)
from ssecast.baselines import build_baselines as build_standard_baselines
from ssecast.metrics import anomaly_correlation_np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate direct 1-14 day multi-horizon SSECast.")
    parser.add_argument("--config", default="config/cascadia.yaml")
    parser.add_argument("--config-name", default="finetune")
    parser.add_argument("--run-name", default="direct14_multihorizon_a")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--high-release-quantile", type=float, default=0.75)
    return parser.parse_args()


def predict(params, checkpoint):
    ds = CascadiaDirectDataset(params, "test")
    loader = DataLoader(
        ds,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=params.device == "cuda",
        num_workers=params.num_workers,
    )
    device = torch.device(params.device)
    model = make_model(params, device)
    load_checkpoint(model, checkpoint, map_location=device)
    model.eval()
    preds, truths, inputs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="direct test"):
            x_dev = x.to(device, non_blocking=True)
            pred, _ = model(x_dev)
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
            inputs.append(x.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(truths, axis=0), np.concatenate(inputs, axis=0), ds


def build_baselines(test_series, horizon):
    release = test_series[:, 0, :]
    n = release.shape[0] - horizon - 1
    base = release[1 : 1 + n]
    prev = release[:n]
    velocity = base - prev
    leads = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]

    out = {}
    out["persistence"] = np.repeat(base[:, None, :], horizon, axis=1)
    out["linear extrapolation"] = base[:, None, :] + velocity[:, None, :] * leads

    alpha, beta = 0.35, 0.08
    level = np.zeros_like(release)
    trend = np.zeros_like(release)
    level[0] = release[0]
    trend[0] = release[1] - release[0]
    for t in range(1, release.shape[0]):
        pred = level[t - 1] + trend[t - 1]
        resid = release[t] - pred
        level[t] = pred + alpha * resid
        trend[t] = trend[t - 1] + beta * resid
    out["Kalman local trend"] = level[1 : 1 + n, None, :] + leads * trend[1 : 1 + n, None, :]
    return out


def compute_metrics(
    pred_norm,
    true_norm,
    input_norm,
    ds,
    train_series,
    analog_history_series,
    high_quantile,
):
    std0 = float(ds.std[0])
    mean0 = float(ds.mean[0])
    pred = pred_norm[:, :, 0, :] * std0 + mean0
    true = true_norm[:, :, 0, :] * std0 + mean0
    prev_true = np.concatenate([input_norm[:, -1:, 0, :], true_norm[:, :-1, 0, :]], axis=1) * std0 + mean0
    true_delta = true - prev_true
    climatology = train_series[:, 0, :].mean(axis=0) * std0 + mean0

    baselines = build_standard_baselines(
        ds.series,
        pred.shape[1],
        train_series,
        analog_history_series=analog_history_series,
    )
    baseline_raw = {name: arr * std0 + mean0 for name, arr in baselines.items()}
    masks = {
        "full field": np.ones_like(true, dtype=bool),
        f"top {int((1.0 - high_quantile) * 100)}% release": high_release_mask(true_delta, high_quantile),
    }

    rows = []
    methods = {"SSECast direct": pred, **baseline_raw}
    for mask_name, mask in masks.items():
        for method, arr in methods.items():
            for h in range(pred.shape[1]):
                m = mask[:, h, :]
                rows.append(
                    {
                        "mask": mask_name,
                        "method": method,
                        "lead_day": h + 1,
                        "NRMSE": weighted_nrmse_np(arr[:, h], true[:, h], m),
                        "ACC": anomaly_correlation_np(
                            arr[:, h],
                            true[:, h],
                            climatology,
                            m,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric(df, metric, mask_name, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    sub = df[df["mask"] == mask_name]
    for method, g in sub.groupby("method"):
        lw = 3.0 if method == "SSECast direct" else 1.8
        alpha = 1.0 if method == "SSECast direct" else 0.75
        ax.plot(g["lead_day"], g[metric], marker="o", linewidth=lw, alpha=alpha, label=method)
    ax.set_xlabel("Forecast lead [day]")
    ax.set_ylabel(metric)
    ax.set_xticks([1, 2, 4, 7, 14, 21, 30])
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    params = load_params(
        args.config,
        args.config_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    params = ensure_direct_params(params, horizon=args.horizon, run_name=args.run_name)
    if args.device is not None:
        params.params["device"] = args.device
        params.device = args.device
    if params.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this process; falling back to CPU.")
        params.params["device"] = "cpu"
        params.device = "cpu"

    checkpoint = args.checkpoint or params.best_checkpoint_path
    out_dir = Path(params.direct_result_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, true, inputs, ds = predict(params, checkpoint)
    train_series = CascadiaDirectDataset(params, "train").series
    eval_series = CascadiaDirectDataset(params, "eval").series
    analog_history_series = np.concatenate([train_series, eval_series], axis=0)
    df = compute_metrics(
        pred,
        true,
        inputs,
        ds,
        train_series,
        analog_history_series,
        args.high_release_quantile,
    )
    csv_path = out_dir / "direct_metrics.csv"
    df.to_csv(csv_path, index=False)
    for mask in df["mask"].unique():
        suffix = mask.replace(" ", "_").replace("%", "pct")
        plot_metric(df, "NRMSE", mask, out_dir / f"{suffix}_nrmse.png")
        plot_metric(df, "ACC", mask, out_dir / f"{suffix}_acc.png")
    print("Saved:", csv_path)
    print("Figures:", out_dir)
    print(df.head())


if __name__ == "__main__":
    main()
