
import torch
from torch import Tensor
from jaxtyping import Float
import DifferentiablePointRender.GaussianSplatting._C as _C


def alpha_blending(
    uv, 
    conic, 
    opacity,
    feature,
    gaussian_ids_sorted,
    title_bins,
    bg, W, H
):
    return _AlphaBlending.apply(
        uv,
        conic,
        opacity,
        feature,
        gaussian_ids_sorted,
        title_bins,
        bg,
        W,
        H
    )


class _AlphaBlending(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        uv, 
        conic,
        opacity,
        feature,
        gaussian_ids_sorted,
        tile_bins,
        bg, 
        W, 
        H):
        
        rendered_feature = _C.alpha_blending_forward(
            uv,
            conic,
            opacity,
            feature,
            gaussian_ids_sorted,
            tile_bins,
            bg,
            W,
            H
        )
        
        # save variables for backward
        # ctx.save_for_backward(
        #     scales, 
        #     uquats, 
        #     visibility_status)
        
        return rendered_feature

    @staticmethod
    def backward(ctx, dL_dtest):
        # get saved variables from forward
        # scales, uquats, visibility_status = ctx.saved_tensors
        
        # (
        #     dL_dscales, 
        #     dL_duquats
        # ) = _C.compute_cov3d_backward(
        #     scales, 
        #     uquats, 
        #     visibility_status, 
        #     dL_dcov3Ds
        # )

        # grads = (
        #     # loss gradient w.r.t scales
        #     dL_dscales,
        #     # loss gradient w.r.t uquats
        #     dL_duquats,
        #     # loss gradient w.r.t visibility_status
        #     None,
        # )

        return None
