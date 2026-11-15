# SSECast — multi-region mixture-of-experts backbone

Source-resolved forecasting of slow-slip evolution from daily geodetic slip-potency
fields. The model advances two consecutive slip-potency fields to a future source
field with an encoder–decoder backbone whose four transformer blocks each contain a
mixture-of-experts (MoE) layer (`expert_num = 4`, `topk = 2`, plus one shared
expert). A single set of shared weights is trained jointly on four subduction zones
— Cascadia, Alaska, Nankai and New Zealand — with region conditioning supplied by a
learnable region-identity embedding, a residual adapter and a region-specific output
head.

```text
SSECast/
├── train.py                     # single-process training (same config; 1 GPU is enough for testing)
├── train_ddp.py                 # multi-GPU DDP training — the reference entry point
├── finetune_per_region.py       # per-region fine-tuning of adapter+head, merged into one checkpoint
├── config/
│   ├── multiregion_l2_ddp.yaml  # training (reference configuration)
│   └── finetune_per_region.yaml # fine-tuning hyper-parameters
├── networks/                    # model definition (SSEPREModel, MoE blocks)
├── utils/                       # data loading, mesh tokenisation, losses, metrics
├── data/<region>/fault/         # fault geometry (included); slip fields are not
└── requirements.txt
```

`multiregion_l2_ddp.yaml` — L2 loss, `rollout_horizon = 2` (two-step roll-out,
`n_blocks = 2`, `block_len = 1`), no Δ (persistence-anchored) parameterisation
(`predict_delta: False`), 4-GPU DDP.

## Installation

Python 3.10+ and a PyTorch build matching the target CPU or GPU.

```bash
pip install -r requirements.txt
```

## Data

**Included.** The fault geometry of all four regions:

```text
data/<region>/fault/{lat, lon, llh, dip, strike, area, height}
```

`lat` and `lon` hold the three vertex coordinates of each triangular element (one row
per element); `dip`, `strike`, `area` and `height` carry the corresponding element
attributes. Element ordering is the ordering used by the slip inversions and is
relied on by the code, so the rows must not be re-ordered.

**Not included.** The slip-potency fields and the normalisation statistics are derived
from geodetic slip inversions and are not distributed with this repository; they are
available from the corresponding author on reasonable request, or can be regenerated
(see *Reconstructing the inputs* below).

Each region needs the following files (the configurations point at `data/<region>/...`
by default):

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
Train, evaluation and test periods must be chronological and non-overlapping. To use a
different location, edit `region_configs.<region>.{data_path, fault_path, llh_path,
mean_path, std_path, tremor_path}` in the configuration.

`slip`, `slip_strike` and `slip_dip` are whitespace-delimited daily arrays of shape
`time × fault element`, in the element order of `fault/lat`.

*Reconstructing the inputs.* We use the near-real-time GNSS position series processed
by Gualandi and co-workers (Gualandi, 2025,
<https://doi.org/10.1093/gji/ggaf198>), distributed through the near-real-time SSE
portal (<https://near-real-time-sse.esc.cam.ac.uk/cascadia/>), and invert them for
daily slip-potency distributions with the PYEQ kinematic inversion software
(<https://github.com/JMNocquet/pyeq36>; Nocquet, 2022,
<https://doi.org/10.5281/zenodo.7467256>). `norm_value/{mean.npy, std.npy}` must be
computed from the training split only, per channel. For a new region, discretise the
Slab2 interface (Hayes et al., 2018, <https://doi.org/10.1126/science.aat4723>) into
triangular elements of ~15 km edge length and keep the element ordering used by the
inversion.

## Training

Reference training run (4 GPUs):

```bash
torchrun --nproc_per_node=4 --standalone train_ddp.py \
  --yaml_config ./config/multiregion_l2_ddp.yaml --config multiregion
```

The same configuration can be used by the single-process trainer, e.g. for a smoke
test on one GPU:

```bash
python train.py --yaml_config ./config/multiregion_l2_ddp.yaml --config multiregion
```

Checkpoints and TensorBoard logs are written to the `exper_path` of the configuration
(`Results/multiregion_l2_ddp/00/...` by default) and are not tracked by Git.

## Fine-tuning

Per-region fine-tuning of the region adapter and output head on top of a trained
backbone, merged back into a single checkpoint (the shared trunk stays
bit-identical):

```bash
python finetune_per_region.py \
  --ft_config l2 --tag all_l2 \
  --ft_regions cascadia,alaska,nankai,newzealand
```

`--base_yaml` defaults to `./config/multiregion_l2_ddp.yaml` (the architecture must
match the pretrained checkpoint), and `config/finetune_per_region.yaml` supplies
`pretrained_ckpt_path`, `out_dir`, learning rate, epochs and patience. All of these can
be overridden on the command line.

The shared trunk is frozen and kept bit-identical; only `region_adapters.<region>` and
`region_heads.<region>` are trained, one region at a time, and the four independently
tuned modules are then merged into a single checkpoint with the same structure as a
training checkpoint (`model_state` / `optimizer_state_dict` / `meta`). The merge step
asserts that every changed tensor belongs to a region-specific module.

| key in `config/finetune_per_region.yaml` | meaning |
| --- | --- |
| `pretrained_ckpt_path` | starting checkpoint; must share the architecture of `base_yaml` |
| `ft_regions` | regions to tune, in execution order (also settable with `--ft_regions`) |
| `keep_adapters` | `true` = adapter + head; `false` = head only (a linear probe) |
| `loss_mode` | fine-tuning loss; `l2` corrects amplitude damping, `l1` matches L1 pre-training |
| `lr`, `max_epochs`, `early_stopping_patience` | budget applied independently to each region |

Do **not** change `regions`, `patch_size`, `token_*` or `pos_embed_mode` in the
fine-tuning configuration: the script loads the checkpoint with `strict=True` and
validates missing/unexpected keys, so an architecture mismatch raises an error.

## Model notes

* Slip-potency fields are z-scored with statistics computed on the training split.
* The irregular triangular mesh is converted to a 1-D token sequence by pairing
  triangles that share their longest edge (`patch_size = 2`), after which the token
  order is chosen to maximise spatial coherence: the native element order is kept
  when it is already coherent (Cascadia, coherence 0.998) and a geometrically sorted
  ("banded") order is used otherwise (Alaska / Nankai / New Zealand, 0.990–0.995).
  See `utils/fault_tokens.py`.
* Blocks 1 and 4 operate on the full token grid; blocks 2 and 3 operate on the
  downsampled half-length sequence, so only blocks 1 and 4 map one-to-one onto fault
  elements.
* `pos_embed_mode: none` — positional information comes from rotary position encoding
  on the token index.
* Region conditioning enters at three places: a learnable region-identity vector added
  to the patch embeddings, a residual adapter after the last block, and a
  region-specific output head.

## Scope

This release contains the training code for the multi-region MoE backbone. Raw GNSS
observations, source inversions (slip-potency fields), normalisation statistics, tremor
catalogues, trained checkpoints and forecast products are not included. No license has been assigned yet; contact the repository
owner before reusing the code.
