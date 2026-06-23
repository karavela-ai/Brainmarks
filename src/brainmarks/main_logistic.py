# This source code is licensed under the Apache License, Version 2.0

import argparse
import datetime
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import torch
import torch.nn as nn
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import brainmarks.utils as ut
import brainmarks.version
from brainmarks.datasets.base import HFDataset
from brainmarks.datasets.registry import create_dataset, list_datasets
from brainmarks.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_logistic.yaml"

METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
    "bacc": sklearn.metrics.balanced_accuracy_score,
    "auc": sklearn.metrics.roc_auc_score,
}

# sklearn scoring names for LogisticRegressionCV
SKLEARN_SCORING = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "bacc": "balanced_accuracy",
    "auc": "roc_auc",
}


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed logistic eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = (
            f"{args.name_prefix}/{args.dataset}__{args.model}__{args.representation}__logistic"
        )
    args.output_dir = f"{args.output_root}/{args.name}"
    output_dir = Path(args.output_dir)

    # remote backup location
    if args.remote_root:
        args.remote_dir = f"{args.remote_root}/{args.name}"
        if S3Path(args.remote_dir).exists():
            ut.rsync(args.remote_dir, args.output_dir)
    else:
        args.remote_dir = None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model logistic probe eval")
    print(f"version: {brainmarks.version.__version__}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating frozen backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))
    backbone.requires_grad_(False)
    backbone.to(device)
    print(f"backbone:\n{backbone}")

    # dataset
    print(f"creating dataset: {args.dataset} ({backbone.__space__})")
    dataset_dict = create_dataset(
        args.dataset, space=backbone.__space__, **(args.dataset_kwargs or {})
    )
    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")
    train_dataset: HFDataset = dataset_dict["train"]
    args.num_classes = train_dataset.num_classes

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    # extract features
    print("extracting features for all splits")
    start_time = time.monotonic()
    features_dict, targets_dict = extract_features(args, backbone, dataset_dict, device)
    extract_time = time.monotonic() - start_time
    print(f"feature extraction time: {datetime.timedelta(seconds=int(extract_time))}")

    for split, features in features_dict.items():
        print(f"{split} features: {features.shape}")

    # combine train and validation since cv is done internally
    # also drop any extra splits
    merged_features_dict = {
        "train": np.concatenate([features_dict["train"], features_dict["validation"]]),
        "test": features_dict["test"],
    }
    merged_targets_dict = {
        "train": np.concatenate([targets_dict["train"], targets_dict["validation"]]),
        "test": targets_dict["test"],
    }
    validation_size = len(features_dict["validation"]) / len(merged_features_dict["train"])

    print("evaluating fixed splits")
    table = evaluate(
        args, merged_features_dict, merged_targets_dict, validation_size=validation_size
    )
    table_fmt = table.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results (fixed splits):\n\n{table_fmt}\n\n")

    if args.n_trials:
        print(f"evaluating random splits (n={args.n_trials})")
        all_features = np.concatenate(list(merged_features_dict.values()))
        all_targets = np.concatenate(list(merged_targets_dict.values()))

        trial_tables = []
        for trial_id in range(1, args.n_trials + 1):
            features_train, features_test, targets_train, targets_test = train_test_split(
                all_features,
                all_targets,
                train_size=len(merged_features_dict["train"]),
                stratify=all_targets,
                random_state=args.seed + trial_id,
            )
            trial_features_dict = {"train": features_train, "test": features_test}
            trial_targets_dict = {"train": targets_train, "test": targets_test}
            trial_table = evaluate(
                args,
                trial_features_dict,
                trial_targets_dict,
                validation_size=validation_size,
                trial_id=trial_id,
            )
            trial_tables.append(trial_table)
            # print test row of table
            print(json.dumps(trial_table.iloc[-1].to_dict()))

        trial_tables = pd.concat(trial_tables, ignore_index=True)
        trial_tables["split"] = pd.Categorical(
            trial_tables["split"], categories=["train", "test"], ordered=True
        )
        summary = (
            trial_tables.groupby(["model", "repr", "clf", "dataset", "split"], observed=True)
            .agg(
                {
                    "trial": "count",
                    "C": ["mean", "std"],
                    **{metric: ["mean", "std"] for metric in args.metrics},
                }
            )
            .reset_index()
        )
        summary.columns = ["model", "repr", "clf", "dataset", "split", "n_trials", "C", "C_std"] + [
            f"{metric}{suffix}" for metric in args.metrics for suffix in ["", "_std"]
        ]
        summary_fmt = summary.to_markdown(index=False, floatfmt=".5g")
        print(f"eval results (random splits):\n\n{summary_fmt}\n\n")
        table = pd.concat([table, trial_tables], ignore_index=True)

    table.to_csv(output_dir / "eval_table.csv", index=False)

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)


def evaluate(
    args: DictConfig,
    features_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    validation_size: float,
    trial_id: int | None = None,
) -> dict:
    random_state = sklearn.utils.check_random_state(args.seed)
    cv_seed = random_state.randint(1000, 10000)

    cv = StratifiedShuffleSplit(
        n_splits=args.cv_folds, test_size=validation_size, random_state=cv_seed
    )
    scoring = SKLEARN_SCORING.get(args.cv_metric, args.cv_metric)
    class_weight = "balanced" if args.balanced_sampling else None

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=args.Cs,
                    cv=cv,
                    scoring=scoring,
                    max_iter=args.max_iter,
                    class_weight=class_weight,
                    n_jobs=args.num_workers,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(features_dict["train"], targets_dict["train"])

    header = {
        "model": args.model,
        "repr": args.representation,
        "clf": "logistic",
        "dataset": args.dataset,
        "trial": trial_id,
        "C": float(clf[-1].C_[0]),
    }

    table = []
    for split in features_dict:
        features = features_dict[split]
        targets = targets_dict[split]

        preds = clf.predict(features)
        pred_scores = None
        if hasattr(clf[-1], "predict_proba"):
            pred_scores = clf.predict_proba(features)[:, 1]
        record = {**header, "split": split}

        bootstrap_result = bootstrap_ci(args, preds, targets, scores=pred_scores)

        for metric in args.metrics:
            metric_fn = METRICS[metric]
            if metric == "auc" and pred_scores is not None:
                record[metric] = metric_fn(targets, pred_scores)
            else:
                record[metric] = metric_fn(targets, preds)
            record[f"{metric}_std"] = bootstrap_result[metric]["std"]
        table.append(record)

    table = pd.DataFrame.from_records(table)
    return table


@torch.inference_mode()
def extract_features(
    args: DictConfig,
    backbone: nn.Module,
    dataset_dict: dict[str, HFDataset],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    backbone.eval()
    print_freq = args.get("print_freq", 20)

    features_dict = {}
    targets_dict = {}

    for split, dataset in dataset_dict.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        metric_logger = ut.MetricLogger(delimiter="  ")
        header = f"extract ({split})"

        all_features = []
        all_targets = []

        for batch in metric_logger.log_every(loader, print_freq, header, len(loader)):
            batch = ut.send_data(batch, device)
            target = batch.pop("target")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                cls_embeds, reg_embeds, patch_embeds = backbone(batch)

            all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
            embeds = all_embeds[args.representation]

            # average over sequence dimension: (n, l, d) -> (n, d)
            if embeds.ndim == 3:
                embeds = embeds.mean(dim=1)

            all_features.append(embeds.cpu().float().numpy())
            all_targets.append(target.cpu().numpy())

        features_dict[split] = np.concatenate(all_features, axis=0)
        targets_dict[split] = np.concatenate(all_targets, axis=0)

    return features_dict, targets_dict


def bootstrap_ci(
    args: DictConfig,
    preds: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray | None = None,
):
    random_state = sklearn.utils.check_random_state(args.seed)

    sample_scores = defaultdict(list)
    for _ in range(500):
        if scores is None:
            preds_, targets_ = sklearn.utils.resample(
                preds, targets, random_state=random_state, stratify=targets
            )
            scores_ = None
        else:
            preds_, targets_, scores_ = sklearn.utils.resample(
                preds, targets, scores, random_state=random_state, stratify=targets
            )
        for metric in args.metrics:
            metric_fn = METRICS[metric]
            if metric == "auc" and scores_ is not None:
                sample_scores[metric].append(metric_fn(targets_, scores_))
            else:
                sample_scores[metric].append(metric_fn(targets_, preds_))

    result = {}
    for metric, values in sample_scores.items():
        result[metric] = {"mean": np.mean(values), "std": np.std(values)}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "dataset",
        type=str,
        help=f"[{', '.join(list_datasets())}]",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model = args.model
    cfg.representation = args.representation
    cfg.dataset = args.dataset
    main(cfg)
