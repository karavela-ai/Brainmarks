import os
from pathlib import Path

from brainmarks.datasets.base import HFDataset, get_dataset_root, load_arrow_dataset
from brainmarks.datasets.registry import register_dataset

CEPHALONAUTS_ROOT = "data/processed"  

def _normalize_subject(subject: str | int) -> str:
    subject = str(subject).strip().lower()
    if subject.startswith("sub-"):
        return subject.replace("-", "")
    if subject.startswith("sub"):
        return subject
    return f"sub{subject}"

def _normalize_space(space: str) -> str:
    return space.strip().lower()


def _create_cephalonauts_one(
    space: str,
    *,
    subject: str | int = "sub0",
    target_key: str | None = None,
    **kwargs,
):
    root = CEPHALONAUTS_ROOT
    subject = _normalize_subject(subject)
    space = _normalize_space(space)

    dataset_dict = {}
    for split in ["train", "validation", "test"]:
        url = f"{root}/cephalonauts_{subject}.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)
        dataset_dict[split] = HFDataset(dataset, target_key=target_key)

    return dataset_dict


@register_dataset
def cephalonauts_one(space: str, subject: str | int = "sub0", **kwargs):
    return _create_cephalonauts_one(space, subject=subject, target_key="text_embedding", **kwargs)


@register_dataset
def cephalonauts_one_sub0(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub0", target_key="text_embedding", **kwargs)


@register_dataset
def cephalonauts_one_sub1(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub1", target_key="text_embedding", **kwargs)


@register_dataset
def cephalonauts_one_sub2(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub2", target_key="text_embedding", **kwargs)


# Short aliases matching the Arrow folder names.
@register_dataset
def cephalonauts_sub0(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub0", target_key="text_embedding", **kwargs)


@register_dataset
def cephalonauts_sub1(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub1", target_key="text_embedding", **kwargs)


@register_dataset
def cephalonauts_sub2(space: str, **kwargs):
    return _create_cephalonauts_one(space, subject="sub2", target_key="text_embedding", **kwargs)
