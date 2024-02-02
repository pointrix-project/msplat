
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import DifferentiablePointRender.GaussianSplatting._C as _C

def ewa_project(
    xyz,
    cov3d,
    viewmat,
    camparam,
    uv,
    w, h,
    visibility_status=None
)->Tuple[Tensor, Tensor, Tensor]:
    
    if visibility_status is None:
        visibility_status = torch.ones_like(uv[:, 0], dtype=torch.bool)
        
    return _EWAProject.apply(
        xyz,
        cov3d,
        viewmat,
        camparam,
        uv,
        w, h,
        visibility_status
    )


class _EWAProject(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz,
        cov3d,
        viewmat,
        camparam,
        uv,
        w, h,
        visibility_status
    ):
        
        (
            conic,
            radii,
            touched_tiles
        ) = _C.ewa_project_forward(
            xyz,
            cov3d,
            viewmat,
            camparam,
            uv,
            w, h,
            visibility_status
        )
        
        ctx.save_for_backward(
            xyz,
            cov3d,
            viewmat,
            camparam,
            radii
        )
        
        return conic, radii, touched_tiles
    
    
    @staticmethod
    def backward(
        ctx,
        dL_dconic,
        dL_dradii,
        dL_dtiles
    ):
        
        (
            xyz,
            cov3d,
            viewmat,
            camparam,
            radii
        ) = ctx.saved_tensors
        
        (
            dL_dxyz,
            dL_dcov3d
        ) = _C.ewa_project_backward(
            xyz,
            cov3d,
            viewmat,
            camparam,
            radii,
            dL_dconic
        )
        
        grads = (
            # loss gradient w.r.t xyz
            dL_dxyz,
            # loss gradient w.r.t cov3d
            dL_dcov3d,
            # loss gradient w.r.t viewmat
            None,
            # loss gradient w.r.t camparam
            None,
            # loss gradient w.r.t uv
            None,
            # loss gradient w.r.t W, 
            None,
            # loss gradient w.r.t H,
            None,
            # loss gradient w.r.t visibility_status
            None,
        )
        
        return grads