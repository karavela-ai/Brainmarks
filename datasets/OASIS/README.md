# OASIS

This folder contains the scripts used to curate OASIS fMRI runs from DeepRep
outputs and build Brainmarks evaluation datasets in Hugging Face Arrow format.

## Contents

- `scripts/identify_runs.py`: scans DeepRep outputs and writes valid runs to
  `metadata/runs.csv`.
- `scripts/make_oasis_splits.py`: creates subject-level train/validation/test
  splits in `metadata/oasis_splits.json`.
- `scripts/make_oasis_eval.py`: converts the split run list into Arrow datasets.
- `metadata/`: local curation outputs. 


## Expected Inputs

`identify_runs.py` expects a DeepRep-style tree with subject/session/ and the cifti and nifti outputs in the func subfolders. For example, the following structure:

```text
<deeprep-root>/
  sub-.../
    ses-.../
      func/
        *_run-*_space-fsLR_den-91k_bold.dtseries.nii
        *_run-*_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz
        *_run-*_space-MNI152NLin6Asym_res-02_desc-preproc_bold.json
```

It keeps runs that have both fsLR CIFTI and MNI NIfTI files, and it filters out
MNI runs with fewer than 100 timepoints.

It also expects a diagnosis CSV , which must include these columns:

```text
Subject, session, diagnosis, changed_CN, sex, Age
```

## Build The Dataset

1. Identify valid runs:

```bash
python datasets/OASIS/scripts/identify_runs.py \
  --deeprep-root /path/to/deeprep \
  --diagnosis-csv /path/to/oasis_diagnosis.csv \
  --out-path datasets/OASIS/metadata/runs.csv
```

2. Create subject-level splits:

```bash
python datasets/OASIS/scripts/make_oasis_splits.py
```

For the cognitively-normal conversion task, use the `--ad-task` flag to create splits that restricts the datasets to CN subjects.

```bash
python datasets/OASIS/scripts/make_oasis_splits.py --ad-task
```

3. Generate Arrow data for a Brainmarks space:

```bash
python datasets/OASIS/scripts/make_oasis_eval.py \
  --space schaefer400
```

Useful options:

- `--space`: any space registered in `brainmarks.readers.READER_DICT`.
- `--max_trs`: number of TRs kept from each run. Defaults to `100`.
- `--tr`: fallback TR written to the dataset. Defaults to `2.0`.
- `--output-root`: output directory. Defaults to
  `datasets/OASIS/data/processed`.
- `--overwrite`: replace an existing Arrow dataset.

The final output is saved as:

```text
datasets/OASIS/data/processed/oasis.schaefer400.arrow/
  train/
  validation/
  test/
```

## Notes

Splits are assigned at the subject level, so all runs for a subject stay in the
same partition. And they are stratified by diagnosis, so the train/validation/test sets have similar distributions of cases and controls.
