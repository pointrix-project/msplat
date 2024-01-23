
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
    depth,
    xy,
    W, H,
    visibility_status
)->Tuple[Tensor, Tensor, Tensor]:
    
    if visibility_status is None:
        visibility_status = torch.ones_like(shs[:, 0, 0], dtype=torch.bool)
        
    return _EWAProject.apply(
        xyz,
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
        xyz,
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
            xyz,
            cov3d,
            viewmat,
            camparam,
            uv,
            depth,
            xy,
            W, H,
            visibility_status
        )
        
        # ctx.W = W
        # ctx.H = H
        # ctx.save_for_backward(
        #     xyz,
        #     cov3d,
        #     viewmat,
        #     projmat,
        #     camparam,
        #     visibility_status
        # )
        
        return (cov2d, touched_tiles, radii)
    
    
    @staticmethod
    def backward(
        ctx,
        dL_dcov2d,
        dL_dopacity
    ):
        
        # W = ctx.W
        # H = ctx.H
        # (
        #     xyz,
        #     cov3d,
        #     viewmat,
        #     projmat,
        #     camparam,
        #     visibility_status
        # ) = ctx.saved_tensors
        
        # (
        #     dL_dxyz,
        #     dL_dcov3d
        # ) = _C.ewa_project_backward(
        #     xyz,
        #     cov3d,
        #     viewmat,
        #     projmat,
        #     camparam,
        #     W, H,
        #     visibility_status,
        #     dL_dcov2d,
        #     dL_dopacity
        # )
        
        grads = (
            # loss gradient w.r.t xyz,
            None,
            # loss gradient w.r.t cov3d,
            None,
            # loss gradient w.r.t viewmat,
            None,
            # loss gradient w.r.t projmat,
            None,
            # loss gradient w.r.t camparam,
            None,
            # loss gradient w.r.t W
            None,
            # loss gradient w.r.t H
            None,
            # loss gradient w.r.t visibility_status
            None,
        )
        
        return grads