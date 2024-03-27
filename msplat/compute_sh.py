import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import msplat._C as _C


def compute_sh(
    shs: Float[Tensor, "P C D"],
    view_dirs: Float[Tensor, "P 3"],
    visible: Bool[Tensor, "P 1"] = None,
) -> Float[Tensor, "P 3"]:
    """
    Compute RGB color from Spherical Harmonics(SHs).

    Parameters
    ----------
    shs : Float[Tensor, "P C D"]
        Spherical Harmonics(SHs).
    view_dirs : Float[Tensor, "P 3"]
        Normalized view direction.
    visible : Bool[Tensor, "P 1"], optional
        The visibility status of each point, by default None

    Returns
    -------
    value : Float[Tensor, "P 3"]
        Value of SH evaluation.
    """
    if visible is None:
        visible = torch.ones_like(shs[:, 0, 0], dtype=torch.bool)

    return _ComputeSH.apply(shs, view_dirs, visible)


class _ComputeSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shs, view_dirs, visible):
        value = _C.compute_sh_forward(shs, view_dirs, visible)

        # save variables for backward
        ctx.save_for_backward(shs, view_dirs, visible)

        return value

    @staticmethod
    def backward(ctx, dL_dvalue):
        # get saved variables from forward
        (shs, view_dirs, visible) = ctx.saved_tensors

        (dL_dshs, dL_dvdirs) = _C.compute_sh_backward(
            shs, view_dirs, visible, dL_dvalue
        )

        grads = (
            # loss gradient w.r.t shs
            dL_dshs,
            # loss gradient w.r.t view_dirs
            dL_dvdirs,
            # loss gradient w.r.t visible
            None,
        )

        return grads
