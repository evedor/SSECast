"""Evaluate a trained SSECast model on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssecast.direct import DirectHorizonDataset, ensure_direct_params, load_checkpoint, load_params, make_model
from ssecast.metrics import anomaly_correlation_np, normalized_rmse_np

COMPONENTS = ("Slip potency", "Slip potency along strike", "Slip potency along dip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a direct multi-horizon SSECast model.")
    parser.add_argument("--config", default="configs/cascadia.yaml")
    parser.add_argument("--config-name", default="backbone")
    parser.add_argument("--run-name", default="ssecast-14")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def predict(params, checkpoint: str):
    dataset = DirectHorizonDataset(params, "test")
    loader = DataLoader(
        dataset, batch_size=params.batch_size, shuffle=False,
        pin_memory=params.device == "cuda", num_workers=params.num_workers,
        persistent_workers=params.num_workers > 0,
    )
    model = make_model(params, torch.device(params.device))
    load_checkpoint(model, checkpoint, map_location=params.device)
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, target in tqdm(loader, desc="test"):
            prediction, _ = model(inputs.to(params.device, non_blocking=True))
            predictions.append(prediction.cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(predictions), np.concatenate(targets), dataset


def to_physical(values: np.ndarray, dataset: DirectHorizonDataset) -> np.ndarray:
    mean = dataset.mean[: values.shape[2]][None, None, :, None]
    std = dataset.std[: values.shape[2]][None, None, :, None]
    return values * std + mean


def calculate_metrics(prediction, target, test_data, train_data) -> pd.DataFrame:
    prediction, target = to_physical(prediction, test_data), to_physical(target, test_data)
    climatology = train_data.raw[:, :target.shape[2], :].mean(axis=0)
    rows = []
    for component, name in enumerate(COMPONENTS[:target.shape[2]]):
        for lead in range(target.shape[1]):
            predicted, observed = prediction[:, lead, component], target[:, lead, component]
            rows.append({
                "component": name,
                "lead_day": lead + 1,
                "NRMSE": normalized_rmse_np(predicted, observed),
                "ACC": anomaly_correlation_np(predicted, observed, climatology[component]),
            })
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    for (name, group), colour in zip(metrics.groupby("component", sort=False), ("#bd1f2d", "#2878b5", "#7651c1")):
        for axis, metric in zip(axes, ("NRMSE", "ACC")):
            axis.plot(group["lead_day"], group[metric], "o-", color=colour, linewidth=2, markersize=4.5, label=name)
            axis.set_ylabel(metric)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("Forecast lead time (days)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    params = load_params(
        args.config, args.config_name, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device,
    )
    params = ensure_direct_params(params, horizon=args.horizon, run_name=args.run_name)
    if args.device is not None:
        params.params["device"], params.device = args.device, args.device
    if params.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this process; falling back to CPU.")
        params.params["device"], params.device = "cpu", "cpu"
    checkpoint = args.checkpoint or params.best_checkpoint_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(params.direct_result_dir) / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction, target, test_data = predict(params, checkpoint)
    metrics = calculate_metrics(prediction, target, test_data, DirectHorizonDataset(params, "train"))
    metrics.to_csv(output_dir / "ssecast_metrics.csv", index=False)
    plot_metrics(metrics, output_dir / "ssecast_metrics.png")
    (output_dir / "test_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
