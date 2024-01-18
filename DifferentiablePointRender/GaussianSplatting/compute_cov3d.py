
import torch
from torch import Tensor
from jaxtyping import Float
import DifferentiablePointRender.GaussianSplatting._C as _C


def compute_cov3d(
    scales: Float[Tensor, "*batch 3"],
    uquats: Float[Tensor, "*batch 4"],
    visibility_status: Float[Tensor, ""] = None,
)->Tensor:
    """Compute the 3D covariance matrix.

    Args:
        scales (Tensor): _description_
        uquats (Tensor): _description_
        visibility_status (Tensor, optional): _description_. Defaults to None.

    Returns:
        - **cov3d** (Tensor): 3D covariance vector, upper right part of the covariance matrix.
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
