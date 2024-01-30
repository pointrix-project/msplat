
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import DifferentiablePointRender.GaussianSplatting._C as _C

def ewa_project(
    cov3d,
    viewmat,
    camparam,
    uv,
    depth,
    xy,
    W, H,
    visibility_status=None
)->Tuple[Tensor, Tensor, Tensor]:
    
    if visibility_status is None:
        visibility_status = torch.ones_like(uv[:, 0], dtype=torch.bool)
        
    return _EWAProject.apply(
        cov3d,
        viewmat,
        camparam,
        uv,
        depth,
        xy,
        W, H,
        visibility_status
    )


class _EWAProject(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cov3d,
        viewmat,
        camparam,
        uv,
        depth,
        xy,
        W, H,
        visibility_status
    ):
        
        (
            cov2d,
            touched_tiles,
            radii
        ) = _C.ewa_project_forward(
            cov3d,
            viewmat,
            camparam,
            uv,
            depth,
            xy,
            W, H,
            visibility_status
        )
        
        ctx.W = W
        ctx.H = H
        ctx.save_for_backward(
            cov3d,
            viewmat,
            camparam,
            uv,
            depth,
            xy,
            visibility_status
        )
        
        return (cov2d, touched_tiles, radii)
    
    
    @staticmethod
    def backward(
        ctx,
        dL_dcov2d,
        dL_dtiles,
        dL_draii
    ):
        W = ctx.W
        H = ctx.H
        (
            cov3d,
            viewmat,
            camparam,
            uv,
            depth,
            xy,
            visibility_status
        ) = ctx.saved_tensors
        
        (
            dL_dcov3d,
            dL_dd
        ) = _C.ewa_project_backward(
            cov3d,
            viewmat,
            camparam,
            uv,
            depth,
            xy,
            W, H,
            visibility_status,
            dL_dcov2d
        )
        
        grads = (
            # loss gradient w.r.t cov3d, √
            dL_dcov3d,
            # loss gradient w.r.t viewmat, √ (future)
            None,
            # loss gradient w.r.t camparam,
            None,
            # loss gradient w.r.t uv,
            None,
            # loss gradient w.r.t depth, √
            dL_dd,
            # loss gradient w.r.t xy,
            None,
            # loss gradient w.r.t W, 
            None,
            # loss gradient w.r.t H,
            None,
            # loss gradient w.r.t visibility_status
            None,
        )
        
        return grads