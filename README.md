# SSECast — multi-region mixture-of-experts backbone

Source-resolved forecasting of slow-slip evolution from daily geodetic slip-potency
fields. The model advances two consecutive slip-potency fields to a future source
field with an encoder–decoder backbone whose four transformer blocks each contain a
mixture-of-experts (MoE) layer (`expert_num = 4`, `topk = 2`, plus one shared
expert). A single set of shared weights is trained jointly on four subduction
zones — Cascadia, Alaska, Nankai and New Zealand — with region conditioning
supplied by a learnable region-identity embedding, a residual adapter and a
region-specific output head.

```text
SSECast/
├── train.py                  # single-GPU / CPU training entry point
├── train_ddp.py              # multi-GPU DDP training (reuses Config from train.py)
├── finetune_per_region.py    # per-region fine-tuning of adapter+head, merged into one checkpoint
├── FINETUNE_PER_REGION.md    # notes on the fine-tuning workflow
├── config/                   # YAML configurations
├── networks/                 # model definition (SSEPREModel, MoE blocks)
├── utils/                    # data loading, mesh tokenisation, losses, metrics
├── data/                     # regional input data (large files, tracked separately)
└── requirements.txt
```

## Installation

Python 3.10+ and a PyTorch build matching the target CPU or GPU.

```bash
pip install -r requirements.txt
```

## Data layout

Each region needs the following files; `config/multiregion*.yaml` point at
`data/<region>/...`.

```text
data/<region>/
├── slip_potency_smooth/
│   ├── train/{slip, slip_strike, slip_dip}
│   ├── eval/{slip, slip_strike, slip_dip}
│   └── test/{slip, slip_strike, slip_dip}
├── fault/{lat, lon, llh, dip, strike, area, height}   # triangular mesh; lat/lon hold three vertices per element
└── norm_value/{mean.npy, std.npy}                     # computed from the training split only
```

`slip` files are whitespace-delimited daily arrays with shape `time × fault element`.
The train, evaluation and test periods must be chronological and non-overlapping.
To use a different location, edit `region_configs.<region>.{data_path, fault_path,
llh_path, mean_path, std_path, tremor_path}` in the YAML.

## Training

Multi-region pre-training:

```bash
python train.py --yaml_config ./config/multiregion_l2.yaml --config multiregion
```

Without the region one-hot embedding (ablation; 1,024 fewer parameters —
67,488,678 instead of 67,489,702):

```bash
python train.py --yaml_config ./config/multiregion_l2_noembed.yaml --config multiregion
```

Multi-GPU with DDP:

```bash
torchrun --nproc_per_node=4 train_ddp.py --yaml_config ./config/multiregion_l2_ddp.yaml --config multiregion
```

Per-region fine-tuning of the region adapter and output head, merged back into a
single checkpoint (the shared trunk stays bit-identical):

```bash
python finetune_per_region.py \
  --base_yaml ./config/multiregion_l2.yaml --base_config multiregion \
  --ft_config l2 --tag all_l2 \
  --ft_regions cascadia,alaska,nankai,newzealand \
  --pretrained_ckpt_path ./Results/multiregion_l2/00/training_checkpoints/best_model.ckpt \
  --out_dir  ./Results/multiregion_l2/finetune_per_region \
  --epochs 20 --patience 5
```

Checkpoints and TensorBoard logs are written under the `exper_path` of each
configuration and are not tracked by Git.

## Configurations

| file | purpose |
| --- | --- |
| `multiregion.yaml` | multi-region, L1 loss |
| `multiregion_l2.yaml` | multi-region, L2 loss (default) |
| `multiregion_l2_ddp.yaml` | DDP variant of the above |
| `multiregion_noembed.yaml`, `multiregion_l2_noembed.yaml` | `region_embed: False` ablations |
| `multiregion_r6_lr*.yaml` | learning-rate sweep |
| `cascadia.yaml`, `alaska.yaml`, `nankai.yaml`, `newzealand.yaml` | single-region training |
| `finetune_per_region.yaml`, `_ddp_smoke.yaml` | fine-tuning / DDP smoke test |

## Model notes

* Slip-potency fields are z-scored with statistics computed on the training split.
* The irregular triangular mesh is converted to a 1-D token sequence by pairing
  triangles that share their longest edge (`patch_size = 2`), after which the token
  order is chosen to maximise spatial coherence: the native element order is kept
  when it is already coherent (Cascadia, coherence 0.998) and a geometrically
  sorted ("banded") order is used otherwise (Alaska / Nankai / New Zealand,
  0.990–0.995). See `utils/fault_tokens.py`.
* Blocks 1 and 4 operate on the full token grid; blocks 2 and 3 operate on the
  downsampled half-length sequence, so only blocks 1 and 4 map one-to-one onto
  fault elements.
* `pos_embed_mode: none` — positional information comes from rotary position
  encoding on the token index.
* Region conditioning enters at three places: a learnable region-identity vector
  added to the patch embeddings, a zero-initialised residual adapter after the last
  block, and a region-specific output head. The `*_noembed` configurations disable
  the first.

## Scope

This release contains the training code for the multi-region MoE backbone. Raw GNSS
observations, source inversions, tremor catalogues, trained checkpoints and forecast
products are not included. No license has been assigned yet; contact the repository
owner before reusing the code.
