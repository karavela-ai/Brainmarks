import torch.nn as nn
from torch import Tensor

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model


class IdentityBackbone(nn.Module):
    __space__: str | None = None

    def extra_repr(self):
        return f"'{self.__space__}'"

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        # get ROI time series, shape [B, T, D]
        roi_time_series = batch["bold"]
        # return as patch embeddings
        return None, None, roi_time_series


@register_model
def identity_schaefer400(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "schaefer400"
    return model

@register_model
def identity_schaefer400_tians3(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "schaefer400_tians3"
    return model

@register_model
def identity_mni(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "mni"
    return model

@register_model
def identity_mni2(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "mni2"
    return model

@register_model
def identity_a424(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "a424"
    return model