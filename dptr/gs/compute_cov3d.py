import torch
from torch import Tensor
from jaxtyping import Float, Bool
import dptr.gs._C as _C


def compute_cov3d(
    scales: Float[Tensor, "P 3"],
    uquats: Float[Tensor, "P 4"],
    visible: Bool[Tensor, "P 1"] = None,
) -> Float[Tensor, "P 6"]:
    """
    Compute the 3D covariance matrix.

    Parameters
    ----------
    scales : Float[Tensor, "P 3"]
        3D scaleing vector for each point.
    uquats : Float[Tensor, "P 4"]
        3D rotations (unit quaternions) for each point.
    visible : Bool[Tensor, "P 1"], optional
        The visibility status of each point, by default None

    Returns
    -------
    cov3d : Float[Tensor, "P 6"]
        The upper-right corner of the 3D covariance matrix, stored in a vector.
    """

    if visible is None:
        visible = torch.ones_like(scales[:, 0], dtype=torch.bool)

    return _ComputeCov3D.apply(scales, uquats, visible)


class _ComputeCov3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scales, uquats, visible):
        cov3Ds = _C.compute_cov3d_forward(scales, uquats, visible)

        # save variables for backward
        ctx.save_for_backward(scales, uquats, visible)

        return cov3Ds

    @staticmethod
    def backward(ctx, dL_dcov3Ds):
        # get saved variables from forward
        scales, uquats, visible = ctx.saved_tensors

        (dL_dscales, dL_duquats) = _C.compute_cov3d_backward(
            scales, uquats, visible, dL_dcov3Ds
        )

        grads = (
            # loss gradient w.r.t scales
            dL_dscales,
            # loss gradient w.r.t uquats
            dL_duquats,
            # loss gradient w.r.t visible
            None,
        )

        return grads
