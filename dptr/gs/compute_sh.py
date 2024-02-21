import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import dptr.gs._C as _C


def compute_sh(
    shs: Float[Tensor, "P D C"],
    degree: int,
    view_dirs: Float[Tensor, "P 3"],
    visible: Bool[Tensor, "P 1"] = None,
) -> Float[Tensor, "P 3"]:
    """
    Compute RGB color from Spherical Harmonics(SHs).

    Parameters
    ----------
    shs : Float[Tensor, "P D C"]
        Spherical Harmonics(SHs).
    degree : int
        The degree of SHs.
    view_dirs : Float[Tensor, "P 3"]
        Normalized view direction.
    visible : Bool[Tensor, "P 1"], optional
        The visibility status of each point, by default None

    Returns
    -------
    rgb_color : Float[Tensor, "P 3"]
        The view-dependent RGB color.
    """
    if visible is None:
        visible = torch.ones_like(shs[:, 0, 0], dtype=torch.bool)

    return _ComputeSH.apply(shs, degree, view_dirs, visible)


class _ComputeSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shs, degree, view_dirs, visible):
        (color, clamped) = _C.compute_sh_forward(shs, degree, view_dirs, visible)

        # save variables for backward
        ctx.degree = degree
        ctx.save_for_backward(shs, view_dirs, visible, clamped)

        return color

    @staticmethod
    def backward(ctx, dL_dcolor):
        # get saved variables from forward
        degree = ctx.degree
        (shs, view_dirs, visible, clamped) = ctx.saved_tensors

        (dL_dshs, dL_dvdirs) = _C.compute_sh_backward(
            shs, degree, view_dirs, visible, clamped, dL_dcolor
        )

        grads = (
            # loss gradient w.r.t shs
            dL_dshs,
            # loss gradient w.r.t dgree
            None,
            # loss gradient w.r.t view_dirs
            dL_dvdirs,
            # loss gradient w.r.t visible
            None,
        )

        return grads
