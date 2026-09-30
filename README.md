# M2M-AGDA

Focused implementation of M2M with accuracy-guided discrete adjustment
(AGDA), plus the post-selection reconstruction attack used to report
normalized reconstruction risk (NRR).

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,experiments]"
```

Run the network-free synthetic example:

```bash
python main.py validate experiments/configs/smoke.yaml
python main.py run experiments/configs/smoke.yaml
```

The synthetic data are generated in memory and the run writes its metrics
under `results/smoke/`. To create a reusable synthetic archive instead:

```bash
python data/generate_synthetic.py
python main.py run experiments/configs/synthetic_npz.yaml
```

### Synthetic NRR example

The model-ablation campaign evaluates reconstruction after AGDA selection on
the shared 10,000-row, 100-feature synthetic dataset:

```bash
python experiments/model_ablation.py \
  --config experiments/configs/model_ablation.yaml \
  --variants m2m \
  --reconstruction-bins 4 \
  --attacker-architecture mlp \
  --attacker-hidden-dim 256 \
  --attacker-second-hidden-dim 128 \
  --attacker-epochs 10 \
  --attacker-learning-rate 0.001 \
  --attacker-batch-size 256 \
  --margin 0.0 \
  --device auto \
  --output-dir results/figure-p-model-ablation
```


## Marketing: raw CSV to NRR

The following is the complete reproduction path for the Marketing example.

### 1. Download and preprocess

```bash
python data/data_preparation.py \
  --dataset marketing \
  --download \
  --seed 0 \
  --output data/prepared/marketing.npz
```

Preprocessing first makes a stratified 60/20/20 train/validation/test split.
Schemas are then fitted on the training split only and reused for validation
and test. Categorical values are mapped to logical category IDs and one-hot
encoded. Dataset recipes with continuous fields use training-fitted quantile
bins.

The command creates:

```text
data/prepared/marketing.npz                 AGDA inputs and labels
data/prepared/marketing.reconstruction.npz  logical reconstruction targets
data/prepared/marketing.metadata.json       source, split, and schema metadata
```

Use `--source path/to/file.csv` instead of `--download` for a local CSV. Add
`--force` when intentionally replacing existing prepared files.

### 2. Validate and run AGDA plus the attacker

The Marketing campaign uses a target accuracy of `0.75`.

```bash
python main.py validate experiments/configs/marketing.yaml 

python experiments/model_ablation.py \
  --config experiments/configs/marketing.yaml \
  --variants m2m \
  --attacker-architecture mlp \
  --attacker-hidden-dim 512 \
  --attacker-second-hidden-dim 256 \
  --attacker-epochs 10 \
  --attacker-learning-rate 0.001 \
  --attacker-batch-size 256 \
  --output-dir results/marketing-example
```

AGDA selects the released representation using validation utility and privacy.
Only after selection does `model_ablation.py` train the reconstruction
attacker and evaluate NRR on the test split. The attacker architecture can
also be `logistic` or `ft-transformer`.

### 3. Read the result

```bash
python - <<'PY'
import csv

with open("results/marketing-example/summary.csv", newline="") as handle:
    result = next(csv.DictReader(handle))
print("test NRR:", result["test_nrr_mean"])
print("test accuracy:", result["test_accuracy_mean"])
PY
```

Detailed run records are in `results/marketing-example/metrics.json`; the
summary table is in `results/marketing-example/summary.csv`.

## Other experiment commands

```bash
# Synthetic leakage versus trace bound
python experiments/leakage_vs_bound.py

# All model-ablation variants and reconstruction NRR
python experiments/model_ablation.py
```

Campaign parameters live in `experiments/configs/`. See `data/README.md` for
every supported dataset recipe.

## Repository layout

```text
main.py                         campaign validation and AGDA entry point
pyproject.toml                  package metadata and dependency groups
M2M_full.pdf                    paper
src/m2m_agda/                   AGDA model, objectives, controller, and runner
src/reconstruct/                post-selection attacker and NRR pipeline
data/data_preparation.py        benchmark download and preprocessing
data/generate_synthetic.py      reproducible synthetic NPZ generator
data/README.md                  dataset contracts and preprocessing details
experiments/leakage_vs_bound.py Synthetic leakage verus trace bound
experiments/model_ablation.py   Model abalation
experiments/configs/            ready-to-run YAML campaigns
```

`pyproject.toml` is required: it declares the runtime dependencies, installs
both packages under `src/`, and provides the optional data and plotting
dependencies used by the commands above.
