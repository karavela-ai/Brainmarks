"""Create Cephalonauts One evaluation datasets in Hugging Face Arrow format.

The script reads Cephalonauts One metadata and fMRI files from the Hugging Face
Hub, builds train/validation/test splits by session, and writes each processed
dataset to:

    datasets/Cephalonauts_One/data/processed/cephalonauts_sub{sub}.{space}.arrow
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from nilearn.signal import clean as clean_signal
from scipy.ndimage import uniform_filter1d
from sentence_transformers import SentenceTransformer

import brainmarks.nisc as nisc
import brainmarks.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parents[3] / "data"
DEFAULT_TR = 0.52
SOURCE_REPO_ID = "karavela/cephalonauts_one"


def make_text_samples(
    transcript: pd.DataFrame,
    timestamps: np.ndarray,
    n_words_context: int = 20,
) -> list[str]:
    offsets = transcript["onset"] + transcript["duration"]
    samples = []
    for timestamp in timestamps:
        mask = offsets <= timestamp
        words = transcript["word"][mask]
        samples.append(" ".join(words[-n_words_context:]))

    return samples


def compute_text_embeddings(texts: list[str], text_model: SentenceTransformer) -> np.ndarray:
    LOGGER.info("Embedding %d text samples", len(texts))
    return text_model.encode(texts, show_progress_bar=True)


def main(args):
    outdir = args.output_root / f"cephalonauts_sub{args.sub}.{args.space}.arrow"
    LOGGER.info("Generating dataset: %s", outdir)
    if outdir.exists() and not args.overwrite:
        LOGGER.warning("Output %s exists; exiting. Use --overwrite to replace.", outdir)
        return 1

    ds = load_dataset(SOURCE_REPO_ID, name="timeseries", split="metadata")
    df = ds.to_pandas().sort_values(["sub", "ses", "run"])
    df = df[df["sub"] == args.sub]

    samples_by_split = {"train": [], "validation": [], "test": []}
    text_model = SentenceTransformer(
        args.text_model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "auto"},
    )

    for _, row in df.iterrows():
        ptid = f"sub-{row['sub']}_ses-{row['ses']}_run-{row['run']}"
        if row["ses"] == args.val_session:
            partition = "validation"
        elif row["ses"] == args.test_session:
            partition = "test"
        else:
            partition = "train"

        if args.space in readers.VOLUME_SPACES:
            path = hf_hub_download(
                repo_id=SOURCE_REPO_ID,
                filename=row["mni_nii"],
                repo_type="dataset",
            )
        else:
            path = hf_hub_download(
                repo_id=SOURCE_REPO_ID,
                filename=row["fsaverage5_hemi-L_gii"],
                repo_type="dataset",
            )
            # Download the right hemisphere file as well, since the reader will need both hemispheres to read the data.
            hf_hub_download(
                repo_id=SOURCE_REPO_ID,
                filename=row["fsaverage5_hemi-R_gii"],
                repo_type="dataset",
            )

        fullpath = Path(path)
        relpath = str(fullpath)

        sample = {
            "sub": row["sub"],
            "visit": row["ses"],
            "run": row["run"],
            "mod": "MR",
            "task": "audiodec",
            "path": relpath,
            "fullpath": str(fullpath),
            "original_tr": row["tr"],
            "ptid": ptid,
            "fmri_offset": row["fmri_offset"],
            "audio_offset": row["audio_offset"],
            "audio_duration": row["audio_duration"],
            "transcript": row["transcript"],
        }
        dest = samples_by_split.get(partition, samples_by_split["train"])
        dest.append(sample)

    for split, samples in samples_by_split.items():
        LOGGER.info("Num samples (%s): %d", split, len(samples))

    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    LOGGER.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "run": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": (
                hfds.Array2D(shape=(None, dim), dtype="float16")
                if args.space != "mni2"
                else hfds.Sequence(hfds.Value("float16"))
            ),
            "mean": (
                hfds.Array2D(shape=(1, dim), dtype="float32")
                if args.space != "mni2"
                else hfds.Sequence(hfds.Value("float16"))
            ),
            "std": (
                hfds.Array2D(shape=(1, dim), dtype="float32")
                if args.space != "mni2"
                else hfds.Sequence(hfds.Value("float16"))
            ),
            "text_embedding": hfds.Sequence(hfds.Value("float32")),
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
                    "args": args,
                    "text_model": text_model,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
                fingerprint=f"cephalonauts-one-sub{args.sub}-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    LOGGER.info("Dataset saved to: %s", outdir)
    return 0


def preprocess_fmri(X: np.ndarray, tr: float, temporal_avg: int = 4) -> np.ndarray:
    X = clean_signal(X, detrend=True, standardize="zscore_sample", t_r=tr)
    X = uniform_filter1d(X, size=temporal_avg, axis=0)

    return X


def generate_samples(samples: list[dict], *, reader, dim: int, args, text_model):
    for sample_info in samples:
        fullpath = sample_info["fullpath"]
        series = reader(fullpath)
        tr = sample_info.get("original_tr", DEFAULT_TR)

        T, D = series.shape
        assert D == dim, f"Path {fullpath} has wrong dimension ({D} != {dim})"

        LOGGER.info("Processing %s with shape (%d, %d)", fullpath, T, D)
        fmri_timestamps = sample_info["fmri_offset"] + np.arange(T) * tr
        transcript = pd.DataFrame(sample_info["transcript"].tolist())
        word_offsets = transcript["onset"] + transcript["duration"]
        timestamps = fmri_timestamps[:: (args.fmri_n_skip + 1)] - args.lag

        mask = timestamps >= sample_info["audio_offset"] + args.audio_context_seconds
        mask = mask & (timestamps <= sample_info["audio_offset"] + sample_info["audio_duration"])
        mask = mask & (timestamps >= word_offsets.nsmallest(args.n_words_context).max())
        mask = mask & (timestamps <= word_offsets.max())
        timestamps = timestamps[mask]

        if args.space == "mni2":
            mean = series.mean(axis=0, keepdims=True)
            std = series.std(axis=0, keepdims=True) + 1e-6
            series = preprocess_fmri(series, tr=tr, temporal_avg=4)
            series = series[:: (args.fmri_n_skip + 1)][mask]

        else:
            series, mean, std = nisc.scale(series)
            series = [
                series[t : t + (args.fmri_n_skip + 1)]
                for t in range(0, T, args.fmri_n_skip + 1)
            ]
            series = [s for s, keep in zip(series, mask) if keep]

        text_samples = make_text_samples(
            transcript=transcript,
            timestamps=timestamps,
            n_words_context=args.n_words_context,
        )
        embeddings_text = compute_text_embeddings(texts=text_samples, text_model=text_model)

        for i in range(len(timestamps)):
            series_i = series[i]
            text_embedding_i = embeddings_text[i]

            yield {
                "sub": sample_info["sub"],
                "visit": sample_info["visit"],
                "run": sample_info["run"],
                "mod": sample_info["mod"],
                "task": sample_info["task"],
                "path": sample_info["path"],
                "start": 0,
                "end": series_i.shape[0],
                "tr": float(sample_info.get("original_tr", DEFAULT_TR)),
                "bold": series_i.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "text_embedding": text_embedding_i.astype(np.float32),
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Cephalonauts One evaluation dataset in Arrow format."
    )
    parser.add_argument("--sub", type=int, default=0, choices=[0, 1, 2], help="Subject number")
    parser.add_argument(
        "--space",
        type=str,
        default="schaefer400",
        choices=list(readers.READER_DICT),
        help="Target anatomical space",
    )
    parser.add_argument("--val-session", type=int, default=4, help="Validation session")
    parser.add_argument("--test-session", type=int, default=7, help="Test session")
    parser.add_argument(
        "--fmri-n-skip",
        type=int,
        default=3,
        help="Number of fMRI TRs to skip between samples",
    )
    parser.add_argument(
        "--lag",
        type=float,
        default=2.5,
        help="Hemodynamic lag, in seconds, subtracted from fMRI timestamps",
    )
    parser.add_argument(
        "--audio-context-seconds",
        type=float,
        default=10.0,
        help="Minimum audio context required for each sample",
    )
    parser.add_argument(
        "--n-words-context",
        type=int,
        default=20,
        help="Number of previous words used as text context",
    )
    parser.add_argument(
        "--text-model-name",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Hugging Face model used for text embeddings",
    )
    parser.add_argument("--num-proc", "--num_proc", dest="num_proc", type=int, default=1)
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory where Arrow dataset folders are written",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument(
        "--writer-batch-size",
        "--writer_batch_size",
        dest="writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size",
    )
    args = parser.parse_args()

    LOGGER.info("Starting Cephalonauts One dataset generation with args: %s", args)
    sys.exit(main(args))
