
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import DifferentiablePointRender.GaussianSplatting._C as _C


def compute_sh(
    shs,
    degree,
    view_dirs,
    visibility_status=None
)->Tensor:
    
    if visibility_status is None:
        visibility_status = torch.ones_like(shs[:, 0, 0], dtype=torch.bool)
    
    return _ComputeSH.apply(
        shs,
        degree,
        view_dirs,
        visibility_status
    )
    

class _ComputeSH(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        shs,
        degree,
        view_dirs,
        visibility_status
    ):
        
        (
            color,
            clamped
        ) = _C.compute_sh_forward(
            shs,
            degree,
            view_dirs,
            visibility_status
        )
        
        # save variables for backward
        ctx.degree = degree
        ctx.save_for_backward(
            shs, 
            view_dirs, 
            visibility_status,
            clamped
        )
        
        return color

    @staticmethod
    def backward(
        ctx, 
        dL_dcolor):
        
        # get saved variables from forward
        degree = ctx.degree
        (
            shs, 
            view_dirs, 
            visibility_status, 
            clamped
        ) = ctx.saved_tensors
        
        (
            dL_dshs, 
            dL_dvdirs
        ) = _C.compute_sh_backward(
            shs,
            degree,
            view_dirs,
            visibility_status,
            clamped,
            dL_dcolor
        )
        
        grads = (
            # loss gradient w.r.t shs
            dL_dshs,
            # loss gradient w.r.t dgree
            None,
            # loss gradient w.r.t view_dirs
            dL_dvdirs,
            # loss gradient w.r.t visibility_status
            None,
        )

        return grads
