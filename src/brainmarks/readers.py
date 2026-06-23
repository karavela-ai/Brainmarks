from typing import Protocol

import numpy as np
import nibabel as nib
from templateflow import api as tflow
from nilearn.image import new_img_like, resample_to_img, smooth_img
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean as clean_signal
from nilearn.datasets import load_mni152_template



from . import nisc


class Reader(Protocol):
    def __call__(self, path: str) -> np.ndarray: ...


def fslr64k_reader() -> Reader:
    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        return series

    return fn


def fslr91k_reader() -> Reader:
    def fn(path: str):
        series = nisc.read_cifti_data(path)
        return series

    return fn


def schaefer400_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer_fslr64k(400)

    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = parcavg(series)
        return series

    return fn


def schaefer400_tians3_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer_tian_fslr91k(400, 3)

    parc_path = nisc.fetch_schaefer_tian(400, 3, space="fslr91k")
    parc_data = nisc.read_cifti_data(parc_path).squeeze(0)
    parc_img = nib.load(parc_path)
    parc_surf_ids, parc_surf_mask = nisc.get_cifti_surf_indices(parc_img)

    def fn(path: str):
        if str(path).endswith(".gii"):
            series_ = nisc.read_gifti_surf_data(path)
            series = np.zeros((len(series_), parc_data.shape[0]), dtype=series_.dtype)
            series[:, parc_surf_ids] = series_[:, parc_surf_mask]
        else:
            series = nisc.read_cifti_data(path)
        series = parcavg(series)
        return series

    return fn


def a424_reader() -> Reader:
    parcavg = nisc.parcel_average_a424()

    parc_path = nisc.fetch_a424(cifti=True)
    parc_data = nisc.read_cifti_data(parc_path).squeeze(0)
    parc_img = nib.load(parc_path)
    parc_surf_ids, parc_surf_mask = nisc.get_cifti_surf_indices(parc_img)

    def fn(path: str):
        if str(path).endswith(".gii"):
            series_ = nisc.read_gifti_surf_data(path)
            series = np.zeros((len(series_), parc_data.shape[0]), dtype=series_.dtype)
            series[:, parc_surf_ids] = series_[:, parc_surf_mask]
        else:
            series = nisc.read_cifti_data(path)
        series = parcavg(series)
        return series

    return fn


def schaefer400_tians3_buckner7_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer400_tians3_buckner7()

    def fn(path: str):
        series = nisc.read_mni152_2mm_data(path, interpolation="linear")
        series = parcavg(series)
        return series

    return fn


def flat_reader() -> Reader:
    resampler = nisc.flat_resampler_fslr64k_224_560()

    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = resampler.transform(series, interpolation="linear")
        series = series[:, resampler.mask_]
        return series

    return fn


def mni_cortex_reader() -> Reader:
    roi_path = nisc.fetch_schaefer(400, space="mni")
    mask = nisc.read_mni152_2mm_data(roi_path, interpolation="nearest") > 0

    def fn(path: str):
        series = nisc.read_mni152_2mm_data(path, interpolation="linear")
        series = series[:, mask]
        return series

    return fn


def mni_reader() -> Reader:
    roi_path = tflow.get(
        "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
    )
    mask = nisc.read_mni152_2mm_data(roi_path, interpolation="nearest") > 0

    def fn(path: str):
        series = nisc.read_mni152_2mm_data(path, interpolation="linear")
        series = series[:, mask]
        return series

    return fn

def mni2_reader() -> Reader:
    masker_path = "resources/tpl-MNI152NLin6Asym_desc-brain_mask.nii.gz"
    masker = nib.load(masker_path)
    masker_data = np.ascontiguousarray(masker.get_fdata().T)

    def fn(path:str):
        img = nib.load(path)
        data = np.ascontiguousarray(img.get_fdata().T)
        result = data[:,masker_data > 0]
        return result
    return fn

# NOTE: this changed from LAS to RAS orientation on 2026-01-14
# MNI derived spaces generated before this date are invalid:
#   - a424 (cifti=False)
#   - schaefer400_tians3_buckner7
#   - mni
#   - mni_cortex
MNI152_2MM_SHAPE = (91, 109, 91)
MNI152_2MM_AFFINE = (
    (2.0, 0.0, 0.0, -90.0),
    (0.0, 2.0, 0.0, -126.0),
    (0.0, 0.0, 2.0, -72.0),
    (0.0, 0.0, 0.0, 1.0),
)


READER_DICT = {
    "fslr64k": fslr64k_reader,
    "fslr91k": fslr91k_reader,
    "schaefer400": schaefer400_reader,
    "schaefer400_tians3": schaefer400_tians3_reader,
    "schaefer400_tians3_buckner7": schaefer400_tians3_buckner7_reader,
    "a424": a424_reader,
    "flat": flat_reader,
    "mni": mni_reader,
    "mni_cortex": mni_cortex_reader,
    "mni2": mni2_reader,
}


DATA_DIMS = {
    "fslr64k": 64984,
    "fslr91k": 91282,
    "schaefer400": 400,
    "schaefer400_tians3": 450,
    "schaefer400_tians3_buckner7": 457,
    "a424": 424,
    "flat": 77763,
    "mni": 228483,
    "mni_cortex": 132032,
    "mni2": 67676,
}


VOLUME_SPACES = {
    "schaefer400_tians3_buckner7",
    "mni",
    "mni_cortex",
    "mni2",
}


CIFTI_SPACES = {
    "schaefer400_tians3",
    "a424",
}
