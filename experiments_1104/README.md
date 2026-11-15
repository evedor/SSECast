# experiments_1104 — multi-region MoE backbone (training code)

Training code for the multi-region SSECast backbone analysed in the accompanying
study: one shared encoder–decoder with four mixture-of-experts (MoE) blocks
(`expert_num = 4`, `topk = 2`, plus one shared expert), trained jointly on four
subduction zones (Cascadia, Alaska, Nankai, New Zealand), with a learnable
region-identity embedding and a region-specific adapter and output head.

This directory is self-contained: run every command **from this directory** so that
`utils.*` and `networks.*` resolve.

```text
experiments_1104/
├── train.py                  # single-GPU / CPU trainer (entry point)
├── train_ddp.py              # multi-GPU DDP trainer (reuses Config from train.py)
├── finetune_per_region.py    # per-region fine-tuning of adapter+head, merged into one checkpoint
├── FINETUNE_PER_REGION.md    # notes on the fine-tuning workflow
├── config/                   # YAML configurations (table below)
├── networks/                 # model definition (SSEPREModel, MoE blocks)
└── utils/                    # data loading, token layout, losses, metrics
```

## Data

Configurations point at `../data/<region>/slip_potency_smooth` (this repository's
`data/` folder). Each region needs:

```text
data/<region>/
├── slip_potency_smooth/
│   ├── train/{slip, slip_strike, slip_dip}
│   ├── eval/{slip, slip_strike, slip_dip}
│   └── test/{slip, slip_strike, slip_dip}
├── fault/{lat, lon, llh}           # triangular mesh; lat/lon hold three vertices per element
└── norm_value/{mean.npy, std.npy}  # computed from the training split only
```

If your data live elsewhere, edit `region_configs.<region>.{data_path,fault_path,llh_path,mean_path,std_path}`
in the YAML. The original experiment tree used `../Data/...`; paths here were
rewritten for this repository.

## Training

Multi-region pre-training (the checkpoint used for the MoE analysis):

```bash
python train.py --yaml_config ./config/multiregion_l2.yaml --config multiregion
```

Without the region one-hot embedding (ablation; 1,024 fewer parameters —
67,488,678 instead of 67,489,702):

```bash
python train.py --yaml_config ./config/multiregion_l2_noembed.yaml --config multiregion
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=4 train_ddp.py --yaml_config ./config/multiregion_l2_ddp.yaml --config multiregion
```

Per-region fine-tuning of the region adapter and output head, merged back into a
single checkpoint (shared trunk stays bit-identical):

```bash
python finetune_per_region.py \
  --base_yaml ./config/multiregion_l2.yaml --base_config multiregion \
  --ft_config l2 --tag all_l2 \
  --ft_regions cascadia,alaska,nankai,newzealand \
  --pretrained_ckpt_path ./Results/multiregion_l2/00/training_checkpoints/best_model.ckpt \
  --out_dir  ./Results/multiregion_l2/finetune_per_region \
  --epochs 20 --patience 5
```

Outputs (checkpoints, TensorBoard logs) go to the `exper_path` of each
configuration. Checkpoints are not tracked by Git.

## Configurations

| file | purpose |
| --- | --- |
| `multiregion.yaml` | multi-region, L1 loss (checkpoint used for the MoE analysis) |
| `multiregion_l2.yaml` | multi-region, L2 loss (current default) |
| `multiregion_l2_ddp.yaml` | DDP variant of the above |
| `multiregion_noembed.yaml`, `multiregion_l2_noembed.yaml` | `region_embed: False` ablations |
| `multiregion_r6_lr*.yaml` | learning-rate sweep |
| `cascadia.yaml`, `alaska.yaml`, `nankai.yaml`, `newzealand.yaml` | single-region training |
| `finetune_per_region.yaml`, `_ddp_smoke.yaml` | fine-tuning / DDP smoke test |

## Implementation notes

* Slip-potency fields are z-scored with statistics computed on the training split only.
* The irregular triangular mesh is converted to a 1-D token sequence by pairing
  triangles that share their longest edge (`patch_size = 2`), after which the token
  order is chosen to maximise spatial coherence: the native element order is kept
  when it is already coherent (Cascadia, 0.998) and a geometrically sorted
  ("banded") order is used otherwise (Alaska / Nankai / New Zealand, 0.990–0.995).
  See `utils/fault_tokens.py`.
* Blocks 1 and 4 operate on the full token grid, blocks 2 and 3 on the downsampled
  half-length sequence; only blocks 1 and 4 therefore map one-to-one onto fault elements.
* `pos_embed_mode: none` — positional information comes from rotary position encoding
  on the token index.
* Region conditioning enters at three places: a learnable region-identity vector added
  to the patch embeddings, a residual adapter after the last block, and a
  region-specific output head. The two `*_noembed` configurations disable the first.
