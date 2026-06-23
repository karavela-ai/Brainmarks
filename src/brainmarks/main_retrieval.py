import argparse
import datetime
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import brainmarks.utils as ut
import brainmarks.version
from brainmarks.classifiers import create_classifier, list_classififiers
from brainmarks.datasets.base import HFDataset
from brainmarks.datasets.registry import create_dataset, list_datasets
from brainmarks.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_retrieval.yaml"

METRICS_DICT = {
    "top1": "Top-1 Retrieval Accuracy (↑)",
    "top10": "Top-10 Retrieval Accuracy (↑)",
    "mrr": "Median Relative Rank (↓)",
    "rank": "Mean Rank (↓)",
    "cosine": "Mean Cosine Similarity (↑)",
}
LOWER_IS_BETTER = {"loss", "mrr", "rank"}


@torch.inference_mode()
def get_embedding_dims(
    args: DictConfig,
    backbone: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    loader = DataLoader(dataset, batch_size=1, collate_fn=ut.collate)
    example_batch = next(iter(loader))
    example_batch = ut.send_data(example_batch, device)
    target_dim = example_batch[args.target_embedding_key].shape[-1]

    print("Bold shape:", example_batch["bold"].shape)
    print("Max value:", example_batch["bold"].mean(dim=1).max().item())
    cls_embeds, reg_embeds, patch_embeds = backbone(example_batch)
    all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
    embeds = all_embeds[args.representation]
    embed_dim = embeds.shape[-1]
    return embed_dim, target_dim

def make_param_groups(model: nn.Module):
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_wd_multiplier = 1.0
        if name.endswith(".bias") or "norm" in name:
            param_wd_multiplier = 0.0
        param_groups.append(
            {
                "params": [param],
                "lr_multiplier": 1.0,
                "wd_multiplier": param_wd_multiplier,
            }
        )
    return param_groups

def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int, no_decay: bool = False):
    warmup = np.linspace(0.0, 1.0, warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 0)
    if not no_decay:
        decay = np.cos(np.linspace(0, np.pi, decay_steps))
        decay = (decay + 1) / 2
    else:
        decay = np.ones(decay_steps)
    lr_schedule = base_lr * np.concatenate([warmup, decay])
    return lr_schedule[:total_steps]

def load_model(args, model, optimizer):
    ckpt_path = Path(args.output_dir) / "checkpoint-last.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        args.start_epoch = ckpt["epoch"] + 1
        meta = ckpt["meta"]
        print(f"loaded model and optimizer state, resuming training from {args.start_epoch}")
    else:
        args.start_epoch = 0
        meta = None

    return meta

def save_model(args, epoch, model, optimizer, meta=None, is_best=None):
    output_dir = Path(args.output_dir)
    last_checkpoint_path = output_dir / "checkpoint-last.pth"
    best_checkpoint_path = output_dir / "checkpoint-best.pth"

    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": OmegaConf.to_container(args),
        "epoch": epoch,
        "meta": meta,
        "is_best": is_best,
    }

    print(f"saving checkpoint {last_checkpoint_path}")
    safe_save(to_save, last_checkpoint_path)
    if is_best:
        print(f"saving best checkpoint {best_checkpoint_path}")
        safe_save(to_save, best_checkpoint_path)

def safe_save(obj, path):
    path = Path(path)
    tmp_path = path.parent / f".tmp-{path.name}"
    torch.save(obj, tmp_path)
    tmp_path.rename(path)


class Retriever(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        representation: str,
        head: nn.Module,
    ):
        super().__init__()
        self.representation = representation
        self.backbone = backbone
        self.head = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cls_embeds, reg_embeds, patch_embeds = self.backbone(batch)
        all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
        embeds = all_embeds[self.representation]
        return self.head(embeds)
    
def make_head(args: DictConfig, embed_dim: int, target_dim: int):
    return create_classifier(
        name=args.classifier,
        in_dim=embed_dim,
        out_dim=target_dim,
        **(args.classifier_kwargs or {}),
    )

def clip_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_one_epoch(
    args: DictConfig,
    model: Retriever,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
):
    model.train()
    if not args.full_finetune:
        model.backbone.eval()
    use_cuda = device.type == "cuda"
    log_wandb = args.wandb and ut.is_main_process()
    print_freq = args.get("print_freq", 100) if not args.debug else 1

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"train: [{epoch}]"
    num_batches = len(data_loader)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, num_batches)
    ):

        batch = ut.send_data(batch, device)
        targets = batch.pop(args.target_embedding_key)

        global_step = epoch * math.ceil(num_batches / args.accum_iter) + (
            batch_idx // args.accum_iter
        )
        need_update = (batch_idx + 1) % args.accum_iter == 0 or (batch_idx + 1) == num_batches
        if need_update:
            lr = lr_schedule[global_step]
            ut.update_lr(optimizer.param_groups, lr)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch)
            loss = clip_loss(pred, targets, temperature=args.temperature)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        (loss / args.accum_iter).backward()

        if need_update:
            total_grad = nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

            log_metric_dict = {
                "lr": lr,
                "loss": loss_value,
                "grad": float(total_grad),
            }
            metric_logger.update(**log_metric_dict)

            if log_wandb:
                wandb.log({f"train/{k}": v for k, v in log_metric_dict.items()}, global_step)

        if use_cuda:
            torch.cuda.synchronize()

    print(f"{header} Summary:", metric_logger)

    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

def retrieval_ranks(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.float()
    targets = targets.float()
    preds_norm = F.normalize(preds, dim=-1)
    targets_norm = F.normalize(targets, dim=-1)
    sim = preds_norm @ targets_norm.T
    correct_sim = sim.diag().unsqueeze(1)
    ranks = (sim > correct_sim).sum(dim=1)
    return ranks


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: Retriever,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
):
    model.eval()
    use_cuda = device.type == "cuda"
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = len(data_loader)
    if args.debug:
        epoch_num_batches = min(epoch_num_batches, 10)

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"eval ({eval_name}): [{epoch}]"

    preds = []
    targets = []
    cosine_sims = []
    losses = []

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)
        target = batch.pop(args.target_embedding_key).detach().float().cpu()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch).detach().float().cpu()

        loss = clip_loss(pred, target, temperature=args.temperature)
        losses.append(loss.item())
        pos_sim = F.cosine_similarity(pred, target, dim=-1)
        cosine_sims.append(pos_sim)
        preds.append(pred)
        targets.append(target)

        if use_cuda:
            torch.cuda.synchronize()

    cosine_sim = torch.cat(cosine_sims).mean().item()
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    total_loss = np.mean(losses)
    stats = {"loss": total_loss, "cosine": cosine_sim}
    metric_values = retrieval_metrics(args, preds, targets)
    for metric, value in metric_values.items():
        stats[metric] = value
    stats = {f"{eval_name}/{k}": v for k, v in stats.items()}
    return stats, preds, targets

def evaluate_checkpoint(
    args: DictConfig,
    model: Retriever,
    eval_loaders_dict: dict[str, DataLoader],
    device: torch.device,
    ckpt_label: str,
    output_dir: Path,
    log_wandb: bool,
):
    ckpt_path = output_dir / f"checkpoint-{ckpt_label}.pth"
    print(f"evaluating {ckpt_label} checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    ckpt_meta = ckpt["meta"]
    print(f"eval model info:\n{json.dumps(ckpt_meta)}")

    header = {
        "model": args.model,
        "repr": args.representation,
        "head": args.classifier,
        "dataset": args.dataset,
        "target_key": args.target_embedding_key,
        "ckpt": ckpt_label,
        "epoch": ckpt_meta["epoch"],
        "lr": args.lr,
        "wd": args.weight_decay,
    }
    eval_stats = {
        f"eval/{ckpt_label}/epoch": header["epoch"],
        f"eval/{ckpt_label}/lr_best": header["lr"],
        f"eval/{ckpt_label}/wd_best": header["wd"],
    }
    table = []

    for split, loader in eval_loaders_dict.items():
        stats, preds, targets = evaluate(
            args,
            model,
            loader,
            args.epochs,
            device,
            eval_name=split,
        )
        record = {**header, "split": split}

        log_prefix = f"eval/{ckpt_label}/{split}"
        record["loss"] = eval_stats[f"{log_prefix}/loss"] = float(
            stats[f"{split}/loss"]
        )
        for metric in args.metrics:
            score = stats[f"{split}/{metric}"]
            record[metric] = eval_stats[f"{log_prefix}/{metric}"] = float(score)

        table.append(record)
        np.savez(output_dir / f"preds_{ckpt_label}_{split}.npz", preds=preds, targets=targets)

    table = pd.DataFrame.from_records(table)
    table.to_csv(output_dir / f"eval_table_{ckpt_label}.csv", index=False)

    with (output_dir / f"eval_log_{ckpt_label}.json").open("w") as f:
        print(json.dumps(eval_stats), file=f)

    if log_wandb:
        wandb.log(eval_stats, args.epochs)

    preferred = "best" if args.get("early_stopping", True) else "last"
    if ckpt_label == preferred:
        table.to_csv(output_dir / "eval_table.csv", index=False)
        eval_stats = {k.replace(f"/{ckpt_label}", ""): v for k, v in eval_stats.items()}
        with (output_dir / "eval_log.json").open("w") as f:
            print(json.dumps(eval_stats), file=f)
        if log_wandb:
            wandb.log(eval_stats, args.epochs)


def parse_k(metric: str) -> int:
    try:
        return int(metric.split("@", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"metric must be formatted as name@k: {metric}") from exc


def retrieval_metrics(
    args: DictConfig,
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    ranks = retrieval_ranks(preds, targets)

    out = {}
    for metric in args.metrics:
        if metric == "cosine":
            continue
        if metric.startswith("top") and metric[3:].isdigit():
            k = int(metric[3:])
            value = (ranks < k).float().mean().item() * 100.0
        elif metric == "mrr":
            value = ((ranks + 1) / len(targets)).median().item()
        elif metric == "rank":
            value = ranks.float().mean().item()
        else:
            raise ValueError(f"unknown retrieval metric: {metric}")
        out[metric] = value
    return out


def is_better_score(metric: str, score: float, best_score: float) -> bool:
    return score < best_score if metric in LOWER_IS_BETTER else score > best_score


def main(args: DictConfig):

    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed retrieval eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = (
            f"{args.name_prefix}/"
            f"{args.dataset}__{args.model}__{args.representation}__{args.classifier}"
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

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model retrieval probe eval")
    print(f"version: {brainmarks.version.__version__}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))

    if args.full_finetune:
        backbone.requires_grad_(True)
    else:
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

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=ut.collate,
    )
    eval_loaders_dict = {}
    for split, dataset in dataset_dict.items():
        eval_loaders_dict[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers else None,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=False,
            collate_fn=ut.collate,
        )
    val_dl = eval_loaders_dict["validation"]
    test_dl = eval_loaders_dict["test"]

    print(
        f"Train samples: {len(train_dl.dataset)} | "
        f"Val samples: {len(val_dl.dataset)} | "
        f"Test samples: {len(test_dl.dataset)}"
    )

    # retrieval head
    print("running backbone on example batch to get dimensions")
    embed_dim, target_dim = get_embedding_dims(args, backbone, train_dataset, device)
    print(f"brain embedding dim ({args.representation}): {embed_dim}")
    print(f"target embedding dim ({args.target_embedding_key}): {target_dim}")

    print("initializing retrieval head")
    head = make_head(args, embed_dim, target_dim)
    model = Retriever(backbone, args.representation, head)
    model.to(device)
    print(f"retrieval head:\n{model.head}")

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    backbone_params_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(
        f"backbone params (train): "
        f"{backbone_params / 1e6:.1f}M ({backbone_params_train / 1e6:.1f}M)"
    )
    head_params = sum(p.numel() for p in model.head.parameters())
    head_params_train = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"head params (train): {head_params / 1e6:.1f}M ({head_params_train / 1e6:.1f}M)")

    # optimizer
    print("setting up optimizer")
    total_batch_size = args.batch_size * args.accum_iter
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum"
    )
    print(f"lr: {args.lr:.2e}")
    param_groups = make_param_groups(model)
    optim_params = sum(p.numel() for group in param_groups for p in group["params"])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"optimizer params: {optim_params / 1e6:.1f}M / "
        f"trainable params: {trainable_params / 1e6:.1f}M"
    )
    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = math.ceil(len(train_dl) / args.accum_iter)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    no_decay = args.get("no_decay", False)
    lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps, no_decay=no_decay)
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps}) (decay = {not no_decay})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    ckpt_meta = load_model(args, model, optimizer)
    initial_best = float("inf") if args.cv_metric in LOWER_IS_BETTER else float("-inf")
    best_score = ckpt_meta["best_score"] if ckpt_meta else initial_best

    print(f"start training for {args.epochs} epochs")
    log_wandb = args.wandb and ut.is_main_process()
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model, 
            train_dl, 
            optimizer, 
            lr_schedule,
            epoch,
            device,
        )
        val_stats, _, _ = evaluate(args, model, val_dl, epoch, device, eval_name="validation")
        val_score = val_stats[f"validation/{args.cv_metric}"]
        hparam_scores = {
            metric: val_stats[f"validation/{metric}"] for metric in ["loss"] + args.metrics
        }
        hparam_scores_fmt = "  ".join(
            f"{METRICS_DICT.get(metric, metric)}: {score:.3f}"
            for metric, score in hparam_scores.items()
        )
        print(f"Validation: [{epoch}]  {hparam_scores_fmt}")

        best_stats = {"train/loss_best": train_stats["train/loss"]}
        for metric, score in hparam_scores.items():
            best_stats[f"validation/{metric}_best"] = score

        if log_wandb:
            wandb.log(best_stats, epoch + 1)

        merged_stats = {"epoch": epoch, **train_stats, **val_stats, **best_stats}
        with (output_dir / "train_log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        is_best = is_better_score(args.cv_metric, val_score, best_score)

        if is_best:
            best_score = val_score
        meta = {
            "score": val_score,
            "epoch": epoch,
            "is_best": is_best,
            "best_score": best_score,
        }

        save_model(
            args,
            epoch,
            model,
            optimizer,
            meta=meta,
            is_best=is_best,
        )

        if args.remote_dir:
            print(f"backing up to remote: {args.remote_dir}")
            ut.rsync(args.remote_dir, output_dir)

        print("-" * 80)

    for ckpt_label in ["last", "best"]:
        evaluate_checkpoint(
            args, model, eval_loaders_dict, device, ckpt_label, output_dir, log_wandb
        )

    table = pd.read_csv(output_dir / "eval_table.csv")
    table_fmt = table.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results:\n\n{table_fmt}\n\n")

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a retrieval model")
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "classifier",
        type=str,
        help=f"[{', '.join(list_classififiers())}]",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=f"[{', '.join(list_datasets())}]",
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))

    cfg.model = args.model
    cfg.representation = args.representation
    cfg.classifier = args.classifier
    cfg.dataset = args.dataset
    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    main(cfg)
