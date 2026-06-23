import json
from pathlib import Path

import fsspec

from brainmarks.datasets.base import HFDataset, get_dataset_root, load_arrow_dataset
from brainmarks.datasets.registry import register_dataset

OASIS_ROOT = "data/processed"  # For local testing with generated Arrow files

def _create_oasis(space: str, target_key: str | None = None, target_map: dict | None = None, **kwargs):
    """Load OASIS Arrow splits for a given anatomical space.
    """

    dataset_dict = {}
    for split in ["train", "validation", "test"]:
        url = f"{OASIS_ROOT}/oasis.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url, **kwargs)

        if target_map is not None:
            dataset = HFDataset(dataset, target_key=target_key, target_map=target_map)
        elif target_key is not None:
            dataset = HFDataset(dataset, target_key=target_key)
        else:
            dataset = HFDataset(dataset)

        dataset_dict[split] = dataset

    return dataset_dict

@register_dataset
def oasis_diagnosis(space: str, **kwargs):
    """OASIS dataset with diagnosis labels as targets."""
    return _create_oasis(space, target_key="diagnosis", **kwargs)

@register_dataset
def oasis_age(space: str, **kwargs):
    """OASIS dataset with age labels as targets."""
    return _create_oasis(space, target_key="age", **kwargs)

@register_dataset
def oasis_sex(space: str, **kwargs):
    """OASIS dataset with sex labels as targets."""
    return _create_oasis(space, target_key="sex", **kwargs)

@register_dataset
def oasis_changed_cn(space: str, **kwargs):
    """OASIS dataset with changed_CN labels as targets."""
    return _create_oasis(space, target_key="changed_CN", **kwargs)
