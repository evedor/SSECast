# SSECast

SSECast is a source-resolved framework for multi-horizon forecasting of slow-slip evolution from daily geodetic source fields. It advances two consecutive slip-potency fields to a sequence of future source fields and supports region-specific training for subduction margins.

This repository is a clean, source-only release prepared from the operational research code. It contains the model, direct multi-horizon training and testing entry points, regional configuration templates and reproducibility notes. Raw GNSS observations, source inversions, tremor catalogues, model checkpoints and forecast products are intentionally excluded.

## Repository layout

```text
SSECast/
├── configs/              # Regional configuration templates
├── data/                 # Local-data specification only; contents are ignored by Git
├── docs/                 # Data and reproducibility notes
├── outputs/              # Local run products; ignored by Git
├── scripts/              # Training and independent-testing commands
└── src/ssecast/          # Model, dataset, forecast objective and metrics
```

## Installation

Create a Python environment with Python 3.10 or later, install PyTorch appropriate for the target CPU or GPU, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data preparation

Place processed regional files under `data/` following [data/README.md](data/README.md). The example configurations retain the regional preprocessing assumptions but point only to local, untracked paths.

## Direct multi-horizon training

```bash
python scripts/train.py \
  --config configs/cascadia.yaml \
  --config-name backbone \
  --run-name ssecast-14 \
  --horizon 14 \
  --device cuda \
  --amp
```

Set `--horizon 30` and use a distinct `--run-name` to train a 30-day model. Training artefacts are written below `outputs/`.

## Independent testing

```bash
python scripts/test.py \
  --config configs/cascadia.yaml \
  --config-name backbone \
  --run-name ssecast-14 \
  --horizon 14 \
  --checkpoint outputs/cascadia/ssecast-14/training_checkpoints/best_model.ckpt \
  --device cuda
```

Testing writes ssecast_metrics.csv and ssecast_metrics.png to the run directory. These report NRMSE and ACC for slip potency, slip potency along strike and slip potency along dip at each forecast lead time. Reference forecasts are not included.

## Reproducible workflow

Prepare the regional files using data/README.md, train with scripts/train.py, select best_model.ckpt using the eval split, and run scripts/test.py once on the held-out test split. See docs/workflow.md for the complete sequence and expected outputs.

## Scope

The current release provides research code and configuration templates. Data availability, scientific interpretation and citation information will be added with the associated manuscript and archival release.

## License

No license has yet been assigned. Please contact the repository owner before reusing the code.
