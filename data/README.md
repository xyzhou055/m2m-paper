# Data contract

Dataset downloads are explicit. Use the built-in synthetic generator, pass
`--download` to a supported preparation recipe, or provide a local CSV.

## NPZ format

An archive must contain exactly these six named arrays:

```text
x_train  y_train
x_val    y_val
x_test   y_test
```

Feature arrays must be finite numeric matrices with shapes `(n_train, d)`,
`(n_val, d)`, and `(n_test, d)` and a common, nonzero `d`. Label arrays must be
one-dimensional integer arrays with the corresponding row counts. Every split
must be nonempty, and every validation/test class must be representable by the
training label vocabulary. Object arrays and pickled objects are not accepted.

The file is a model-ready boundary. Before writing it:

1. make the train/validation/test split, preferably stratified for
   classification;
2. fit imputation, encoding, scaling, or vocabulary construction on training
   rows only;
3. apply that fitted transformation to validation and test rows; and
4. store dense numeric features and integer class labels.

Do not fit preprocessing on the concatenated data. The loader validates and
casts arrays but deliberately does not infer a benchmark-specific preprocessing
pipeline.

Configure the NPZ data source and path in a YAML campaign. Keep large or
restricted archives outside version control; the repository ignores local NPZ
payloads by default.

## Synthetic data

The synthetic source is ready in two forms and requires no network access.

The standard smoke campaign generates the data in memory:

```bash
python main.py run experiments/configs/smoke.yaml
```

All bundled synthetic campaign files, and the generator defaults below, use
10,000 samples, 100 features, and 50 relevant features.

To create a reusable archive in this directory, run:

```bash
python data/generate_synthetic.py
```

This writes `data/synthetic_ready.npz`. The matching campaign can then validate
and train from that file:

```bash
python main.py validate experiments/configs/synthetic_npz.yaml
python main.py run experiments/configs/synthetic_npz.yaml
```

Use `python data/generate_synthetic.py --help` to change the seed, dimensions,
class count, split fractions, or destination. Generated NPZ files are ignored
by Git.

### Synthetic preprocessing

The generator performs these operations in order:

1. sample class labels and independent Gaussian input features;
2. add class-dependent mean shifts to the configured relevant features and,
   if requested, inject label noise;
3. create stratified train, validation, and test indices;
4. compute each feature's mean and standard deviation from training rows only;
5. replace near-zero training standard deviations with one, then apply the
   training mean and scale to all three splits; and
6. save contiguous `float32` feature matrices and `int64` label vectors using
   the six-array NPZ contract above.

Splitting before standardization prevents validation/test statistics from
leaking into model fitting. The synthetic data are intended for tests, smoke
runs, and controlled sensitivity checks, not as a replacement for a paper
benchmark.

## Preprocessing an external dataset

Use the following leakage-safe sequence before writing an NPZ archive:

1. define the prediction label and remove identifiers or columns that would
   not be available at deployment time;
2. split raw rows into train, validation, and test sets, stratifying by class
   when possible;
3. fit missing-value rules, categorical vocabularies/encoders, clipping
   thresholds, and numeric scaling on the training split only;
4. apply those fitted transformations unchanged to validation and test rows,
   including an explicit unknown-category policy;
5. map labels to contiguous integer IDs using the training label vocabulary
   and reject validation/test labels absent from training; and
6. verify finite dense matrices, aligned row counts, a shared feature order,
   and the exact six required array names before saving.

The AGDA loader does not learn or repair preprocessing. This keeps covariance,
feature weights, controller decisions, and final test evaluation on a clearly
separated protocol.

## Adult, CoverType, Marketing, Mushroom, and NATICUSdroid

These datasets preserve every source row and share stratified splitting, label
encoding, training-only schema fitting, and artifact writing. They are never
subsampled or class-rebalanced, but their feature recipes are not identical:

- Adult keeps its original class distribution, treats known or numeric columns
  as continuous, converts them to training-fitted quantile bins, and one-hot
  encodes all resulting logical features.
- CoverType downloads the complete OpenML dataset `1596`. Numeric columns with
  more than 10 training values are quantile-binned; its binary indicator
  columns are treated as categorical and one-hot encoded.
- Mushroom keeps the full dataset, treats every input as categorical, and
  fits each vocabulary on training rows.
- Marketing extracts `Marketing/raw/raw.csv` from the CleanML 2020 dataset
  archive, predicts `Income`, and treats all 13 input attributes as logical
  categorical features. The seed-0 preparation has 83 one-hot columns.
- NATICUSdroid validates native binary indicators, removes columns that are
  constant in training, and keeps the remaining binary matrix directly.

Install the optional preprocessing dependencies:

```bash
python -m pip install -e ".[data]"
```

Preparing a local CSV requires the dataset name and source. The target column
is inferred from known names or can be supplied explicitly:

```bash
python data/data_preparation.py \
  --dataset mushroom \
  --source raw/mushroom.csv \
  --target-column class
```

Downloads are never implicit. Pass `--download` to fetch Adult, CoverType, or
Mushroom from OpenML, Marketing from CleanML, or NATICUSdroid from UCI:

```bash
python data/data_preparation.py --dataset adult --download
python data/data_preparation.py --dataset covertype --download
python data/data_preparation.py --dataset marketing --download
python data/data_preparation.py --dataset mushroom --download
python data/data_preparation.py --dataset naticusdroid --download
```

`covtype` and `coverypte` are accepted as aliases for `covertype`. Regardless
of the alias, outputs use the canonical `covertype` name.

Each invocation writes three aligned files under `data/prepared/` unless
`--output` is provided:

```text
adult.npz                 six model arrays used by AGDA
adult.reconstruction.npz  logical categorical IDs and category counts
adult.metadata.json       source, split, schema, and feature mappings
```

When a model-ablation campaign points to the model NPZ, `model_ablation.py`
automatically verifies the metadata fingerprint and uses the reconstruction
sidecar for exact logical-feature NRR. An NPZ without these sidecars remains
supported and falls back to training-fitted categorical/quantile targets.
