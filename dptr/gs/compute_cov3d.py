
import torch
from torch import Tensor
from jaxtyping import Float, Bool
import dptr.gs._C as _C


def compute_cov3d(
    scales: Float[Tensor, "P 3"],
    uquats: Float[Tensor, "P 4"],
    visibility_status: Bool[Tensor, "P 1"] = None,
)->Float[Tensor, "P 6"]:
    """
    Compute the 3D covariance matrix.

    Parameters
    ----------
    scales : Float[Tensor, "P 3"]
        3D scaleing vector for each point.
    uquats : Float[Tensor, "P 4"]
        3D rotations (unit quaternions) for each point.
    visibility_status : Bool[Tensor, "P 1"], optional
        The visibility status of each point, by default None

    Returns
    -------
    cov3d : Float[Tensor, "P 6"]
        The upper-right corner of the 3D covariance matrices, stored in a vector.
    """
    
    if visibility_status is None:
        visibility_status = torch.ones_like(scales[:, 0], dtype=torch.bool)
    
    return _ComputeCov3D.apply(
        scales,
        uquats,
        visibility_status
    )


class _ComputeCov3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        scales, 
        uquats, 
        visibility_status):
        
        cov3Ds = _C.compute_cov3d_forward(
            scales, 
            uquats, 
            visibility_status
        )
        
        # save variables for backward
        ctx.save_for_backward(
            scales, 
            uquats, 
            visibility_status)
        
        return cov3Ds

    @staticmethod
    def backward(ctx, dL_dcov3Ds):
        # get saved variables from forward
        scales, uquats, visibility_status = ctx.saved_tensors
        
        (
            dL_dscales, 
            dL_duquats
        ) = _C.compute_cov3d_backward(
            scales, 
            uquats, 
            visibility_status, 
            dL_dcov3Ds
        )

        grads = (
            # loss gradient w.r.t scales
            dL_dscales,
            # loss gradient w.r.t uquats
            dL_duquats,
            # loss gradient w.r.t visibility_status
            None,
        )

        return grads
