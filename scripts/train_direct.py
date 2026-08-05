from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ssecast.direct import (
    CascadiaDirectDataset,
    direct_horizon_loss,
    ensure_direct_params,
    load_one_step_backbone_except_head,
    load_params,
    make_model,
)


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Direct 1-14 day multi-horizon training for SSECast.")
    parser.add_argument("--config", default="config/cascadia.yaml")
    parser.add_argument("--config-name", default="finetune")
    parser.add_argument("--run-name", default="direct14_multihorizon_a")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-quantile", type=float, default=0.75)
    parser.add_argument("--active-weight", type=float, default=3.0)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--aux-loss-coef", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument(
        "--pretrained-one-step",
        default="",
        help="Optional one-step SSECast checkpoint. Matching backbone weights are loaded; the 30-day head is new.",
    )
    return parser.parse_args()


def run_epoch(model, loader, params, device, args, optimizer=None, scheduler=None, scaler=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    batches = 0
    pbar = tqdm(loader, desc=desc)
    use_amp = bool(args.amp and device.type == "cuda")

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_seq, aux_loss = model(x)
                loss = direct_horizon_loss(
                    pred_seq,
                    y,
                    x,
                    active_quantile=args.active_quantile,
                    active_weight=args.active_weight,
                    delta_weight=args.delta_weight,
                )
                if aux_loss is not None:
                    loss = loss + float(args.aux_loss_coef) * aux_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total += float(loss.detach().cpu())
        batches += 1
        pbar.set_postfix(loss=total / max(batches, 1))
    return total / max(batches, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    params = load_params(
        args.config,
        args.config_name,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
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

    result_dir = Path(params.direct_result_dir)
    ckpt_dir = result_dir / "training_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(params.writer).mkdir(parents=True, exist_ok=True)
    with open(result_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = CascadiaDirectDataset(params, "train")
    eval_ds = CascadiaDirectDataset(params, "eval")
    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=params.device == "cuda",
        num_workers=params.num_workers,
        persistent_workers=params.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=params.device == "cuda",
        num_workers=params.num_workers,
        persistent_workers=params.num_workers > 0,
    )

    device = torch.device(params.device)
    model = make_model(params, device)
    if args.pretrained_one_step:
        info = load_one_step_backbone_except_head(model, args.pretrained_one_step, map_location=device)
        print(f"Loaded {len(info[loaded])} tensors from one-step checkpoint; skipped {len(info[skipped])} tensors.")
    else:
        print("Training direct 1-14 day model from scratch: no pretrained checkpoint was loaded.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=0.01, eps=1e-6)
    total_steps = params.max_epochs * len(train_loader)
    scheduler = cosine_warmup_scheduler(optimizer, max(1, int(total_steps * 0.05)), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    writer = SummaryWriter(params.writer)

    best_eval = float("inf")
    history = []
    last_ckpt = None
    t0 = time.time()
    for epoch in range(params.max_epochs):
        train_loss = run_epoch(
            model, train_loader, params, device, args, optimizer, scheduler, scaler, f"train {epoch + 1}/{params.max_epochs}"
        )
        with torch.no_grad():
            eval_loss = run_epoch(model, eval_loader, params, device, args, desc=f"eval {epoch + 1}/{params.max_epochs}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/eval_direct", eval_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "eval_loss_direct14": eval_loss}
        history.append(row)
        print(row)

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "params": params.params,
            "args": vars(args),
        }
        last_ckpt = ckpt
        torch.save(ckpt, params.checkpoint_path)
        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(ckpt, params.best_checkpoint_path)
            print("Saved best checkpoint:", params.best_checkpoint_path)
        with open(result_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    if last_ckpt is not None:
        torch.save(last_ckpt, params.final_checkpoint_path)
        print("Final checkpoint:", params.final_checkpoint_path)
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")
    print("Best checkpoint:", params.best_checkpoint_path)
    print("Last checkpoint:", params.final_checkpoint_path)


if __name__ == "__main__":
    main()
