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
├── FINETUNE_PER_REGION.md       # notes on the fine-tuning workflow
├── config/
│   ├── multiregion_l2_ddp.yaml          # training  (reference configuration)
│   ├── multiregion_l2_ddp_noembed.yaml  # ablation: no region one-hot embedding
│   └── finetune_per_region.yaml         # fine-tuning hyper-parameters
├── networks/                    # model definition (SSEPREModel, MoE blocks)
├── utils/                       # data loading, mesh tokenisation, losses, metrics
├── data/                        # regional input data
└── requirements.txt
```

Only one training configuration is maintained: `multiregion_l2_ddp.yaml` — L2 loss,
`rollout_horizon = 2` (two-step roll-out, `n_blocks = 2`, `block_len = 1`), no Δ
(persistence-anchored) parameterisation (`predict_delta: False`), 4-GPU DDP. The
accompanying ablation switches the region one-hot embedding off and is otherwise
identical.

## Installation

Python 3.10+ and a PyTorch build matching the target CPU or GPU.

```bash
pip install -r requirements.txt
```

## Data layout

Each region needs the following files; the configurations point at `data/<region>/...`.

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

Ablation without the region one-hot embedding (1,024 fewer parameters — 67,488,678
instead of 67,489,702):

```bash
torchrun --nproc_per_node=4 --standalone train_ddp.py \
  --yaml_config ./config/multiregion_l2_ddp_noembed.yaml --config multiregion
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

`--base_yaml` defaults to `./config/multiregion_l2_ddp.yaml` (architecture must match
the pretrained checkpoint), and `config/finetune_per_region.yaml` supplies
`pretrained_ckpt_path`, `out_dir`, learning rate, epochs and patience. All of these
can be overridden on the command line.

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
  to the patch embeddings, a zero-initialised residual adapter after the last block,
  and a region-specific output head. The `*_noembed` configuration disables the first.

## Scope

This release contains the training code for the multi-region MoE backbone. Raw GNSS
observations, source inversions, tremor catalogues, trained checkpoints and forecast
products are not included. No license has been assigned yet; contact the repository
owner before reusing the code.
