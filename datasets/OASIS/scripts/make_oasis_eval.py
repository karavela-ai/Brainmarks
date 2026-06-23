"""Create OASIS evaluation dataset (Arrow format).

This script creates HuggingFace Arrow datasets for OASIS fMRI using the
previously-generated `oasis_splits.json` file in `datasets/OASIS/metadata/`.

It expects each entry in the JSON to contain:
- PTID: "sub-XX_ses-YY_run-Z"
- Partition: one of "train", "validation", "test"
- dtseries_path: full path to fsLR dtseries (CIFTI)
- mni_path: full path to MNI nifti (volume)

The script picks the appropriate file per `--space` and writes Arrow datasets
to `datasets/OASIS/data/processed/oasis.{space}.arrow`.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np

import brainmarks.nisc as nisc
import brainmarks.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).parents[1]
OASIS_METADATA = DATASET_ROOT / "metadata" / "oasis_splits.json"
TARGETS = ["diagnosis", "changed_CN", "sex", "Age"]

# Defaults
DEFAULT_TR = 2.0
MAX_TRS = 100


def parse_ptid(ptid: str) -> tuple[str, str]:
    """Parse PTID values formatted as `sub-XX_ses-YY_run-Z`."""
    parts = ptid.split("_")
    sub = parts[0] if parts else ""
    ses = parts[1] if len(parts) > 1 else ""
    return sub, ses


def main(args):
    outdir = args.output_root / f"oasis.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists() and not args.overwrite:
        _logger.warning("Output %s exists; exiting. Use --overwrite to replace.", outdir)
        return 1

    # Load metadata JSON
    with OASIS_METADATA.open() as f:
        curation_data = json.load(f)
    _logger.info("Loaded %d samples from OASIS splits JSON", len(curation_data))

    # Build samples by split
    samples_by_split = {"train": [], "validation": [], "test": []}

    for entry in curation_data:
        ptid = entry.get("PTID")
        partition = entry.get("Partition", "train")
        dtseries = entry.get("dtseries_path")
        mni = entry.get("mni_path")

        sub, ses = parse_ptid(ptid)

        if args.space in readers.VOLUME_SPACES:
            fullpath = Path(mni)
            relpath = str(fullpath)
        else:
            fullpath = Path(dtseries)
            relpath = str(fullpath)

        sample = {
            "sub": sub,
            "visit": ses,
            "mod": "MR",
            "task": "rest",
            "path": relpath,
            "fullpath": str(fullpath),
            "original_tr": float(args.tr),
            "ptid": ptid,
        }
        for target_name in TARGETS:
            sample[target_name] = entry.get(target_name)
        dest = samples_by_split.get(partition, samples_by_split["train"])
        dest.append(sample)

    for split, samples in samples_by_split.items():
        _logger.info("Num samples (%s): %d", split, len(samples))

    # Load reader for target space
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    # Features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
            **{target: hfds.Value("string") for target in TARGETS},
        }
    )

    writer_batch_size = args.writer_batch_size
    if writer_batch_size is None:
        if args.space == "flat":
            writer_batch_size = 16
        elif args.space in {"mni", "mni_cortex"}:
            writer_batch_size = 8

    cache_root = Path(args.cache_dir) if args.cache_dir else outdir.parent / ".hf-cache"
    cache_root.mkdir(exist_ok=True, parents=True)

    # Generate datasets
    with tempfile.TemporaryDirectory(dir=cache_root, prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, samples in samples_by_split.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "samples": samples,
                    "reader": reader,
                    "dim": dim,
                    "max_trs": args.max_trs,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
                fingerprint=f"oasis-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    _logger.info("Dataset saved to: %s", outdir)
    return 0


def generate_samples(samples: list[dict], *, reader, dim: int, max_trs: int):
    for sample_info in samples:
        fullpath = sample_info["fullpath"]

        series = reader(fullpath)

        T, D = series.shape
        assert D == dim, f"Path {fullpath} has wrong dimension ({D} != {dim})"

        end = min(T, max_trs)
        series = series[:end]
        series, mean, std = nisc.scale(series)

        yield {
            "sub": sample_info["sub"],
            "visit": sample_info["visit"],
            "mod": sample_info["mod"],
            "task": sample_info["task"],
            "path": sample_info["path"],
            "start": 0,
            "end": end,
            "tr": float(sample_info.get("original_tr", DEFAULT_TR)),
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            **{
                target: None if sample_info.get(target) is None else str(sample_info.get(target))
                for target in TARGETS
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create OASIS evaluation dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="schaefer400",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: schaefer400)",
    )
    parser.add_argument(
        "--num_proc",
        "-j",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size (default: 16 for flat, otherwise datasets default)",
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Directory for Hugging Face generator cache",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATASET_ROOT / "data" / "processed",
        help="Directory where Arrow dataset folders are written",
    )
    parser.add_argument(
        "--overwrite",
        "-x",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument("--max_trs", type=int, default=MAX_TRS, help="Number of TRs to keep")
    parser.add_argument("--tr", type=float, default=DEFAULT_TR, help="Assumed TR for OASIS data")
    args = parser.parse_args()
    sys.exit(main(args))
