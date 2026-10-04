# SSECast

SSECast is a source-resolved framework for multi-horizon forecasting of slow-slip evolution from daily geodetic source fields. It advances two consecutive slip-potency fields to future source fields and supports independent training for individual subduction margins.

It includes the model, direct multi-horizon training and testing scripts, and configuration templates. Raw GNSS observations, source inversions, tremor catalogues, trained checkpoints and forecast products are not concluded.

## Repository layout

```text
SSECast/
├── configs/              # Regional training configurations
├── data/                 # Local input data
├── outputs/              # Local checkpoints and evaluation products
├── scripts/              # Training and held-out testing entry points
└── src/ssecast/          # Model, dataset, loss and metrics
```

## Installation

Use Python 3.10 or later. Install a PyTorch build appropriate for the target CPU or GPU, then install the remaining requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data layout

The regional configuration `configs/cascadia.yaml` expects the following files for Cascadia. The other regions use the same layout.

```text
data/
  cascadia/
    slip_potency_smooth/
      train/{slip,slip_strike,slip_dip}
      eval/{slip,slip_strike,slip_dip}
      test/{slip,slip_strike,slip_dip}
    norm_value/{mean.npy,std.npy}
```

Each source-field file is a whitespace-delimited daily array with shape `time × fault element`. The train, evaluation and test periods must be chronological and non-overlapping. Compute the normalization arrays from the training split only.

Data should be prepared using the preprocessing code available from https://github.com/Geolandi/sse_postprocessing (Julia) or https://github.com/Geolandi/sse_postprocessing_matlab (MATLAB). Note that the MATLAB implementation requires a separate implementation of the filtering step. The corresponding data should then be downloaded from https://near-real-time-sse.esc.cam.ac.uk/cascadia/, processed with the selected workflow, and organized into the data layout described above.

The data in the repository were processed through September 2025.

## Training

Each configuration has one selectable entry: `backbone`. It describes the first, from-scratch SSECast training stage; there is no separate fine-tuning configuration.

Train a 14-day Cascadia model with:

```bash
python scripts/train.py \
  --config configs/cascadia.yaml \
  --config-name backbone \
  --run-name ssecast-14 \
  --horizon 14 \
  --device cuda \
  --amp
```

Train a 30-day model with a separate run name and horizon:

```bash
python scripts/train.py \
  --config configs/cascadia.yaml \
  --config-name backbone \
  --run-name ssecast-30 \
  --horizon 30 \
  --device cuda \
  --amp
```

SSECast-14 and SSECast-30 are independently trained direct multi-horizon models. The 30-day model is not obtained by recursively extending the 14-day model. Training uses the evaluation split to select `best_model.ckpt` and writes checkpoints, TensorBoard logs, `train_history.json` and `train_args.json` below `outputs/<region>/<run-name>/`.

## testing

Evaluate the selected model once on the untouched test split:

```bash
python scripts/test.py \
  --config configs/cascadia.yaml \
  --config-name backbone \
  --run-name ssecast-14 \
  --horizon 14 \
  --checkpoint outputs/cascadia/ssecast-14/training_checkpoints/best_model.ckpt \
  --device cuda
```

Testing never changes model weights. It writes `ssecast_metrics.csv`, `ssecast_metrics.png` and `test_args.json` to the run directory. The metrics report normalized root-mean-square error and anomaly correlation coefficient for slip potency, slip potency along strike and slip potency along dip at every forecast lead time. Test examples are not used for normalization, model selection or optimization.

## Daily forecast results
The daily forecast results for the Cascadia region can be viewed from the website below
https://ssecast-cascadia.github.io/



## Scope and license

This release provides research code and configuration templates. Data availability, scientific interpretation and citation information will accompany the manuscript and archival release. No license has yet been assigned; contact the repository owner before reusing the code.
