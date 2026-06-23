import argparse
import json
import logging
import re
from pathlib import Path

import nibabel as nib
import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGETS = ["diagnosis", "changed_CN", "sex", "Age"]
MIN_TIMEPOINTS = 100


def find_valid_runs(func_dir: str | Path) -> dict[str, dict[str, str | float | None]]:
    """Find runs with both fsLR CIFTI and MNI NIfTI files."""
    func_dir = Path(func_dir)
    if not func_dir.exists():
        LOGGER.warning("Missing func directory: %s", func_dir)
        return {}

    runs = {}

    for path in func_dir.iterdir():
        match = re.search(r"run-(\d+)", path.name)
        if not match:
            continue

        run_id = match.group(1)
        runs.setdefault(run_id, {"dtseries": None, "mni": None, "TR": None})

        if "space-MNI152NLin6Asym_res-02_desc-preproc_bold.json" in path.name:
            with path.open() as f:
                metadata = json.load(f)
            runs[run_id]["TR"] = metadata.get("RepetitionTime")

        if "space-fsLR_den-91k_bold.dtseries.nii" in path.name:
            runs[run_id]["dtseries"] = str(path)
        if "space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz" in path.name:
            img = nib.load(path)
            if img.shape[-1] >= MIN_TIMEPOINTS:
                runs[run_id]["mni"] = str(path)
            else:
                LOGGER.info(
                    "Skipping run %s in %s: only %d timepoints",
                    run_id,
                    func_dir,
                    img.shape[-1],
                )

    return {
        run_id: paths
        for run_id, paths in runs.items()
        if paths["dtseries"] is not None and paths["mni"] is not None
    }


def find_targets(
    subject: str,
    session: str,
    diagnosis_df: pd.DataFrame,
    targets: list[str],
) -> dict[str, object]:
    rows = diagnosis_df.loc[
        (diagnosis_df["Subject"].astype(str) == subject)
        & (diagnosis_df["session"].astype(str) == session)
    ]
    if rows.empty:
        raise ValueError(f"No diagnosis row found for subject={subject}, session={session}")

    targets_dict = {}
    for target in targets:
        targets_dict[target] = rows.iloc[0][target]
    return targets_dict


def main(deeprep_root: str | Path, out_path: str | Path, diagnosis_csv: str | Path):
    deeprep_root = Path(deeprep_root)
    out_path = Path(out_path)
    diagnosis_df = pd.read_csv(diagnosis_csv)
    rows = []

    subjects = sorted(deeprep_root.glob("sub-*"))
    for subject in subjects:
        sessions = sorted(subject.glob("ses-*"))
        for session in sessions:
            runs = find_valid_runs(session / "func")
            if not runs:
                continue

            targets = find_targets(subject.name[4:], session.name[4:], diagnosis_df, TARGETS)
            for run, paths in runs.items():
                row = {
                    "subject": subject.name,
                    "session": session.name,
                    "run": run,
                    "dtseries_path": paths["dtseries"],
                    "mni_path": paths["mni"],
                    "TR": paths["TR"],
                }
                row.update(targets)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOGGER.info("Wrote %d valid OASIS runs to %s", len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify valid runs in OASIS dataset and save to CSV."
    )
    parser.add_argument(
        "--deeprep-root",
        type=str,
        required=True,
        help="Path to the root of the DeepRep output directory.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="datasets/OASIS/metadata/runs.csv",
        help="Path to save the CSV file with valid runs.",
    )
    parser.add_argument(
        "--diagnosis-csv",
        type=str,
        required=True,
        help="CSV containing diagnosis and target columns for each subject/session.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s]: %(message)s",
        level=logging.INFO,
        datefmt="%y-%m-%d %H:%M:%S",
    )
    main(args.deeprep_root, args.out_path, args.diagnosis_csv)
