"""Create subject-level OASIS train/validation/test splits.

The input is `datasets/OASIS/metadata/runs.csv`, produced by
`identify_runs.py`. The output is `datasets/OASIS/metadata/oasis_splits.json`,
which is consumed by `make_oasis_eval.py`.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

OASIS_ROOT = Path(__file__).parents[1]
TARGETS = ["diagnosis", "changed_CN", "sex", "Age"]
LOGGER = logging.getLogger(__name__)


def _labels_for_subjects(subjects: list[str], subject_labels: dict[str, object]) -> list[object]:
    return [subject_labels[subject] for subject in subjects]


def main():
    parser = argparse.ArgumentParser(description="Create OASIS subject-level splits.")
    parser.add_argument(
        "--ad-task",
        "--ad_task",
        dest="ad_task",
        action="store_true",
        help="Restrict to cognitively normal subjects and stratify by changed_CN.",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=OASIS_ROOT / "metadata" / "runs.csv",
        help="CSV produced by identify_runs.py.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=OASIS_ROOT / "metadata" / "oasis_splits.json",
        help="Destination JSON path.",
    )
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)

    if args.ad_task:
        runs = runs[runs["diagnosis"] == "Cognitively normal"]
        stratify_column = "changed_CN"
    else:
        stratify_column = "diagnosis"

    subject_df = (
        runs[["subject", stratify_column]]
        .dropna()
        .drop_duplicates(subset=["subject"], keep="first")
    )
    subject_df["subject"] = subject_df["subject"].astype(str)
    subjects = subject_df["subject"].tolist()
    subject_labels = dict(zip(subject_df["subject"], subject_df[stratify_column], strict=True))

    train_subjects, heldout_subjects = train_test_split(
        subjects,
        test_size=0.3,
        random_state=42,
        stratify=_labels_for_subjects(subjects, subject_labels),
    )
    val_subjects, test_subjects = train_test_split(
        heldout_subjects,
        test_size=0.5,
        random_state=42,
        stratify=_labels_for_subjects(heldout_subjects, subject_labels),
    )

    splits = {
        "train": set(train_subjects),
        "validation": set(val_subjects),
        "test": set(test_subjects),
    }

    splits_list = []
    for row in runs.itertuples():
        subject = str(row.subject)
        session = row.session
        run = row.run
        dtseries_path = row.dtseries_path
        mni_path = row.mni_path
        partition = None
        for split_name, split_subjects in splits.items():
            if subject in split_subjects:
                partition = split_name
                break

        if partition is None:
            continue

        splits_list.append({
            "PTID": f"{subject}_{session}_run-{run}",
            "Partition": partition,
            "dtseries_path": dtseries_path,
            "mni_path": mni_path,
            "TR": row.TR,
        })
        for target_name in TARGETS:
            splits_list[-1][target_name] = getattr(row, target_name)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        json.dump(splits_list, f, indent=2)

    LOGGER.info(
        "Wrote %d OASIS run records to %s (%d train, %d validation, %d test subjects)",
        len(splits_list),
        args.out_path,
        len(train_subjects),
        len(val_subjects),
        len(test_subjects),
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s]: %(message)s",
        level=logging.INFO,
        datefmt="%y-%m-%d %H:%M:%S",
    )
    main()
