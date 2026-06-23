"""Run logistic probes for several models and plot test metrics.

Example:
    python -m brainmarks.compare_logistic oasis_changed_cn cortex_mae_flat brainlm_vitmae_111m \
        --config src/brainmarks/config/compare_logistic.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from brainmarks.datasets.registry import list_datasets
from brainmarks.models.registry import list_models

plt.style.use("seaborn-v0_8-paper")

DATASET_LEGEND_DICT = {
    "oasis_changed_cn": "OASIS-3 (future dementia)",
}

REPRESENTATION_DATA_DICT = {
    "parcel": "o",
    "flat": "^",
    "volume": "s",
}


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run brainmarks.main_logistic for several models and compare metrics."
    )
    parser.add_argument("dataset", type=str, help=f"Dataset name. Available: {list_datasets()}")
    parser.add_argument("models", nargs="+", help=f"Model names. Available: {list_models()}")
    parser.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent / "config" / "compare_logistic.yaml",
        help="Comparison YAML containing representations, overrides, and plotting settings.",
    )
    return parser.parse_args()


def load_compare_config(path: str | Path) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True) or {}


def resolve_config_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    package_relative = Path(__file__).parent / path
    if package_relative.exists():
        return package_relative
    return path


def to_dotlist(overrides) -> list[str]:
    if not overrides:
        return []
    if isinstance(overrides, dict):
        return [f"{key}={value}" for key, value in overrides.items()]
    return list(overrides)


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def is_same_run_config(run_cfg, output_dir: Path) -> bool:
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        return False

    try:
        previous_cfg = OmegaConf.load(config_path)
    except Exception as exc:
        print(f"Could not load existing config {config_path}: {exc}")
        return False

    current = OmegaConf.to_container(run_cfg, resolve=True)
    previous = OmegaConf.to_container(previous_cfg, resolve=True)
    return current == previous


def finalize_run_config(run_cfg, model: str, representation: str, dataset: str):
    run_cfg.model = model
    run_cfg.representation = representation
    run_cfg.dataset = dataset
    run_cfg.distributed = False

    if not run_cfg.get("name"):
        run_cfg.name = f"{run_cfg.name_prefix}/{dataset}__{model}__{representation}__logistic"
    run_cfg.output_dir = f"{run_cfg.output_root}/{run_cfg.name}"

    if run_cfg.remote_root:
        run_cfg.remote_dir = f"{run_cfg.remote_root}/{run_cfg.name}"
    else:
        run_cfg.remote_dir = None

    return run_cfg


def select_values(
    table: pd.DataFrame,
    metrics: list[str],
    split: str = "test",
    summary_source: str = "auto",
) -> pd.DataFrame:
    selected = table[table["split"] == split].copy()
    if selected.empty:
        print(f"No rows found for split={split!r}")
        return selected

    missing_metrics = [metric for metric in metrics if metric not in selected.columns]
    if missing_metrics:
        raise ValueError(f"missing metric columns in eval table: {missing_metrics}")

    if summary_source == "auto":
        trial_rows = selected[selected["trial"].notna()]
        selected = trial_rows if not trial_rows.empty else selected[selected["trial"].isna()]
    elif summary_source == "trials":
        selected = selected[selected["trial"].notna()]
    elif summary_source == "fixed":
        selected = selected[selected["trial"].isna()]
    elif summary_source != "all":
        raise ValueError(f"unknown summary_source={summary_source!r}")

    return selected[metrics]


def summarize_metrics(values: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for metric in values.columns:
        summary[f"{metric}_mean"] = values[metric].mean()
        summary[f"{metric}_std"] = values[metric].std(ddof=0)
    return summary


def plot_results(
    results_df: pd.DataFrame,
    metric_names: list[str],
    title: str,
    output_path: str | Path,
    models_dict: dict,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(6 * n_metrics, 4),
        sharex=True,
        constrained_layout=False,
    )
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, len(results_df)))
    offsets = np.linspace(-0.4, 0.4, len(results_df))
    model_handles = []
    model_labels = []

    for metric_idx, metric_name in enumerate(metric_names):
        ax = axes[metric_idx]

        for row_idx, (_, row) in enumerate(results_df.iterrows()):
            model_cfg = models_dict.get(row["model"], {})
            marker = REPRESENTATION_DATA_DICT.get(model_cfg.get("type", "parcel"), "o")
            ax.errorbar(
                x=offsets[row_idx],
                y=row[f"{metric_name}_mean"],
                yerr=row[f"{metric_name}_std"],
                fmt=marker,
                markersize=8,
                capsize=4,
                elinewidth=1.5,
                capthick=1,
                color=colors[row_idx],
            )

            if metric_idx == 0:
                model_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        linestyle="None",
                        markerfacecolor=colors[row_idx],
                        markeredgecolor=colors[row_idx],
                        markersize=8,
                    )
                )
                model_labels.append(row["model"])

        ax.set_xlabel(metric_name)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(-0.5, 0.5)

    fig.suptitle(title, y=0.98)
    models_legend = fig.legend(
        model_handles,
        model_labels,
        loc="upper left",
        bbox_to_anchor=(0.05, -0.02, 0.78, 0.1),
        mode="expand",
        ncol=min(len(model_labels), 4),
        frameon=False,
        title="Models",
    )
    fig.add_artist(models_legend)

    representation_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
        )
        for marker in REPRESENTATION_DATA_DICT.values()
    ]
    fig.legend(
        representation_handles,
        list(REPRESENTATION_DATA_DICT),
        loc="upper left",
        bbox_to_anchor=(0.85, -0.02, 0.2, 0.1),
        ncol=1,
        frameon=False,
        title="Representations",
    )
    fig.subplots_adjust(bottom=0.24, top=0.85, wspace=0.35)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = build_parser()
    cfg = load_compare_config(args.config)
    logistic_config = cfg.get("logistic_config")
    models_dict = cfg.get("models", {})
    metrics = as_list(cfg.get("metrics", None))
    split = cfg.get("eval_split", "test")

    if not metrics:
        raise ValueError("No metrics configured. Set metrics in the comparison config.")

    failures = []
    results_rows = []
    default_logistic_config = Path(__file__).parent / "config" / "default_logistic.yaml"

    for model in args.models:
        model_cfg = models_dict.get(model, {})
        representation = model_cfg.get("representation", cfg.get("default_representation", "cls"))

        run_overrides = []
        for key in ("output_root", "name_prefix"):
            if cfg.get(key) is not None:
                run_overrides.append(f"{key}={cfg[key]}")
        run_overrides.extend(to_dotlist(cfg.get("overrides", {})))
        run_overrides.extend(to_dotlist(model_cfg.get("overrides", {})))

        run_cfg = OmegaConf.load(default_logistic_config)
        if logistic_config:
            run_cfg = OmegaConf.unsafe_merge(
                run_cfg,
                OmegaConf.load(resolve_config_path(logistic_config)),
            )
        if run_overrides:
            run_cfg = OmegaConf.unsafe_merge(run_cfg, OmegaConf.from_dotlist(run_overrides))

        run_cfg = finalize_run_config(run_cfg, model, representation, args.dataset)
        output_dir = Path(run_cfg.output_dir)

        if (
            cfg.get("skip_existing", False)
            and output_dir.exists()
            and is_same_run_config(run_cfg, output_dir)
        ):
            print(f"Skipping existing run: {output_dir}")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                run_config_path = Path(tmp_dir) / "logistic_run.yaml"
                OmegaConf.save(run_cfg, run_config_path)
                cmd = [
                    sys.executable,
                    "-m",
                    "brainmarks.main_logistic",
                    model,
                    representation,
                    args.dataset,
                    "--config",
                    str(run_config_path),
                ]
                print(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"Error running command for model {model}: {exc}")
                    failures.append(model)
                    continue

        results_path = output_dir / "eval_table.csv"
        if not results_path.exists():
            print(f"Results file not found for model {model}: {results_path}")
            failures.append(model)
            continue

        results = pd.read_csv(results_path)
        values = select_values(
            results,
            metrics=metrics,
            split=split,
            summary_source=cfg.get("summary_source", "auto"),
        )
        if values.empty:
            print(f"No usable rows found for model {model} in {results_path}")
            failures.append(model)
            continue

        results_rows.append(
            {
                "model": model,
                "repr": representation,
                "dataset": args.dataset,
                "split": split,
                **summarize_metrics(values),
                "n": len(values),
            }
        )

    results_df = pd.DataFrame(results_rows)
    if results_df.empty:
        raise RuntimeError(f"No logistic results collected. Failures: {failures}")

    output_root = Path(cfg.get("output_root", "output"))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / cfg.get("summary_name", "logistic_comparison.csv")
    plot_path = output_root / cfg.get("plot_name", "logistic_comparison_plot.png")

    results_df.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(results_df.to_markdown(index=False, floatfmt=".5g"))
    print(f"\nSaved summary: {summary_path}")

    plot_results(
        results_df,
        metrics,
        f"Logistic Probe Comparison on {DATASET_LEGEND_DICT.get(args.dataset, args.dataset)}",
        output_path=plot_path,
        models_dict=models_dict,
    )
    print(f"Saved plot: {plot_path}")

    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
