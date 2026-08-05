from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PretrainedConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def install_dependency_fallbacks():
    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.PretrainedConfig = PretrainedConfig
        sys.modules["transformers"] = transformers_mod

    try:
        import timm  # noqa: F401
    except Exception:
        timm_mod = types.ModuleType("timm")
        timm_models_mod = types.ModuleType("timm.models")
        timm_layers_mod = types.ModuleType("timm.models.layers")

        class DropPath(torch.nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                mask.floor_()
                return x.div(keep_prob) * mask

        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        timm_layers_mod.DropPath = DropPath
        timm_layers_mod.trunc_normal_ = trunc_normal_
        timm_models_mod.layers = timm_layers_mod
        timm_mod.models = timm_models_mod
        sys.modules["timm"] = timm_mod
        sys.modules["timm.models"] = timm_models_mod
        sys.modules["timm.models.layers"] = timm_layers_mod


install_dependency_fallbacks()

from ssecast.model import SSEPREModel  # noqa: E402


class DirectConfig(PretrainedConfig):
    """SSECast backbone with a direct H-day output head."""

    def __init__(
        self,
        params,
        hidden_size=256,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=1024,
        mlp_bias=True,
        dropout=0.05,
        expert_num=4,
        topk=2,
        output_router_logits=True,
        aux_loss_coef=0.01,
        **kwargs,
    ):
        self.params = params
        self.horizon = int(params.horizon)
        self.fea_len = params.fea_len
        self.time_chans = params.time_chans
        self.data_chans = params.data_chans
        self.patch_size = params.patch_size
        self.out_chans = int(params.data_chans) * self.horizon
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = True
        self.attention_bias = False
        self.max_seq_len = params.fea_len / params.patch_size
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)


class DirectHorizonModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.horizon = int(params.horizon)
        self.data_chans = int(params.data_chans)
        self.backbone = SSEPREModel(DirectConfig(params))

    def forward(self, x):
        raw, aux = self.backbone(x)
        bsz, channels, n_patch = raw.shape
        expected = self.horizon * self.data_chans
        if channels != expected:
            raise RuntimeError(f"Expected {expected} output channels, got {channels}.")
        return raw.reshape(bsz, self.horizon, self.data_chans, n_patch), aux


def _set_param(params, key, value):
    params.params[key] = value
    setattr(params, key, value)


def load_params(config_path: str, config_name: str, **overrides):
    with open(config_path, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f)
    params = SimpleNamespace(params={})
    for key, value in all_params[config_name].items():
        if value == "None":
            value = None
        _set_param(params, key, value)
    for key, value in overrides.items():
        if value is not None:
            _set_param(params, key, value)
    return params


def ensure_direct_params(params, horizon=14, run_name="direct14_multihorizon_a"):
    updates = {
        "horizon": int(horizon),
        "run_num": run_name,
        "out_chans": int(params.data_chans) * int(horizon),
        "num_workers": int(getattr(params, "num_workers", 2)),
    }
    updates["direct_result_dir"] = str(Path(params.exper_path) / run_name)
    updates["checkpoint_path"] = str(Path(updates["direct_result_dir"]) / "training_checkpoints" / "backbone.ckpt")
    updates["best_checkpoint_path"] = str(Path(updates["direct_result_dir"]) / "training_checkpoints" / "best_model.ckpt")
    updates["final_checkpoint_path"] = str(Path(updates["direct_result_dir"]) / "training_checkpoints" / "final_model.ckpt")
    updates["writer"] = str(Path(updates["direct_result_dir"]) / "tensorboard")
    for key, value in updates.items():
        _set_param(params, key, value)
    return params


class CascadiaDirectDataset(Dataset):
    """Input two cumulative fields; target is the next H cumulative fields."""

    def __init__(self, params, split: str):
        self.params = params
        self.split = split
        self.horizon = int(params.horizon)
        self.dir = Path(params.data_path) / split

        slip = np.loadtxt(self.dir / "slip")
        slip_strike = np.loadtxt(self.dir / "slip_strike")
        slip_dip = np.loadtxt(self.dir / "slip_dip")
        fields = [slip, slip_strike, slip_dip]

        if getattr(params, "use_tremor", False):
            tremor = np.loadtxt(Path(params.tremor_path) / split / "tremor_density")
            if split == "train":
                tremor_len = tremor.shape[0]
                fields = [arr[-tremor_len:] for arr in fields]
            fields.append(tremor)

        self.raw = np.stack(fields, axis=1).astype(np.float32)
        self.mean = np.load(params.mean_path).astype(np.float32)[: self.raw.shape[1]]
        self.std = np.load(params.std_path).astype(np.float32)[: self.raw.shape[1]]
        self.series = ((self.raw - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)
        if self.series.shape[0] <= self.horizon + 1:
            raise ValueError(f"{split} split is too short for horizon={self.horizon}")

    def __len__(self):
        return self.series.shape[0] - self.horizon - 1

    def __getitem__(self, index):
        x = self.series[index : index + 2]
        y = self.series[index + 2 : index + 2 + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_model(params, device):
    model = DirectHorizonModel(params).to(device)
    return model


def load_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=strict)
    return ckpt


def load_one_step_backbone_except_head(model, checkpoint_path, map_location="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    new_state = model.state_dict()
    loaded, skipped = [], []
    for key, value in state.items():
        target_key = f"backbone.{key}" if not key.startswith("backbone.") else key
        if target_key in new_state and new_state[target_key].shape == value.shape:
            new_state[target_key] = value
            loaded.append(target_key)
        else:
            skipped.append(target_key)
    model.load_state_dict(new_state, strict=True)
    return {"loaded": loaded, "skipped": skipped, "checkpoint": str(checkpoint_path)}


def direct_horizon_loss(
    pred_seq,
    true_seq,
    input_seq,
    release_channel=0,
    active_quantile=0.75,
    active_weight=3.0,
    delta_weight=0.5,
    eps=1e-8,
):
    """Cumulative H-step loss plus true release-increment loss.

    The active mask is based on true_delta = P(t+h)-P(t+h-1). The first-step
    delta uses x[:, -1] as the previous state, so h=1 is included.
    """

    prev_true = torch.cat([input_seq[:, -1:].detach(), true_seq[:, :-1].detach()], dim=1)
    true_delta = true_seq[:, :, release_channel, :] - prev_true[:, :, release_channel, :]
    q = torch.quantile(true_delta.abs().detach(), active_quantile, dim=-1, keepdim=True)
    spatial_weight = 1.0 + active_weight * (true_delta.abs() >= q).float()
    spatial_weight = spatial_weight.unsqueeze(2)

    state_err = spatial_weight * (pred_seq - true_seq).pow(2)
    state_ref = spatial_weight * true_seq.pow(2)
    state_loss = torch.sqrt(state_err.sum() / (state_ref.sum() + eps))

    prev_pred = torch.cat([input_seq[:, -1:].detach(), pred_seq[:, :-1]], dim=1)
    pred_delta = pred_seq - prev_pred
    delta_err = spatial_weight * (pred_delta - (true_seq - prev_true)).pow(2)
    delta_ref = spatial_weight * (true_seq - prev_true).pow(2)
    inc_loss = torch.sqrt(delta_err.sum() / (delta_ref.sum() + eps))
    return state_loss + float(delta_weight) * inc_loss


def weighted_nrmse_np(pred, true, mask=None, eps=1e-12):
    if mask is None:
        mask = np.ones_like(true, dtype=bool)
    w = mask.astype(np.float64)
    return float(np.sqrt(np.sum(w * (pred - true) ** 2) / (np.sum(w * true**2) + eps)))


def pcc_np(pred, true, mask=None, eps=1e-12):
    if mask is not None:
        pred = pred[mask]
        true = true[mask]
    pred = pred.reshape(-1) - pred.mean()
    true = true.reshape(-1) - true.mean()
    return float(np.sum(pred * true) / (np.sqrt(np.sum(pred**2) * np.sum(true**2)) + eps))


def high_release_mask(true_delta, quantile=0.75):
    threshold = np.quantile(np.abs(true_delta), quantile, axis=-1, keepdims=True)
    return np.abs(true_delta) >= threshold
