# Cephalonauts One

This folder contains the script used to build Brainmarks evaluation datasets
from the Cephalonauts One Hugging Face dataset (`karavela/cephalonauts_one`).
The generated Arrow datasets pair fMRI samples with text embeddings from the
preceding transcript context.

## Contents

- `scripts/make_eval.py`: builds one subject/space Arrow dataset.

## Requirements

Authenticate with Hugging Face to download the Cephalonauts One dataset:

```bash
huggingface-cli login
```

The default text model is `Qwen/Qwen3-Embedding-8B`, which is large and should
usually be run on a GPU machine.

## Build An Evaluation Dataset

Generate a dataset for one subject and one target space:

```bash
python datasets/Cephalonauts_One/scripts/make_eval.py \
  --sub 0 \
  --space schaefer400
```

Useful options:

- `--sub`: subject ID, one of `0`, `1`, or `2`.
- `--space`: any space registered in `brainmarks.readers.READER_DICT`.
- `--val-session` and `--test-session`: sessions assigned to validation and
  test splits.
- `--fmri-n-skip`: temporal stride. The default `3` keeps one sample every four
  TRs.
- `--text-model-name`: Hugging Face embedding model.
- `--output-root`: output directory. Defaults to
  `data/processed`.
- `--overwrite`: replace an existing Arrow dataset.

The command writes a Hugging Face `DatasetDict` saved to disk:

```text
data/processed/cephalonauts_sub0.schaefer400.arrow/
  train/
  validation/
  test/
```