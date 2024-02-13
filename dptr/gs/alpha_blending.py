
import torch
from torch import Tensor
from jaxtyping import Float
import dptr.gs._C as _C


def alpha_blending(
    uv, 
    conic, 
    opacity,
    feature,
    gaussian_idx_sorted,
    title_bins,
    bg, W, H
):
    return _AlphaBlending.apply(
        uv,
        conic,
        opacity,
        feature,
        gaussian_idx_sorted,
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
        gaussian_idx_sorted,
        tile_bins,
        bg, 
        W, 
        H):
        
        (
            render_feature, 
            final_T, 
            ncontrib
        ) = _C.alpha_blending_forward(
            uv,
            conic,
            opacity,
            feature,
            gaussian_idx_sorted,
            tile_bins,
            bg,
            W,
            H
        )
        
        ctx.W = W
        ctx.H = H
        ctx.bg = bg
        ctx.save_for_backward(
            uv,
            conic,
            opacity,
            feature,
            gaussian_idx_sorted,
            tile_bins,
            final_T,
            ncontrib
        )
        
        return render_feature, final_T, ncontrib

    @staticmethod
    def backward(ctx, dL_drendered, dL_dT, dL_dncontrib):
        W = ctx.W
        H = ctx.H
        bg = ctx.bg
        
        (
            uv,
            conic,
            opacity,
            feature,
            gaussian_idx_sorted,
            tile_bins,
            final_T,
            ncontrib
        ) = ctx.saved_tensors
        
        (
            dL_duv,
            dL_dconic,
            dL_dopacity,
            dL_dfeature
        ) = _C.alpha_blending_backward(
            uv,
            conic,
            opacity,
            feature,
            gaussian_idx_sorted,
            tile_bins,
            bg,
            W,
            H,
            final_T,
            ncontrib,
            dL_drendered
        )
        
        grads = (
            # grads w.r.t uv
            dL_duv,
            # grads w.r. conic,
            dL_dconic,
            # grads w.r. opacity,
            dL_dopacity,
            # grads w.r. feature,
            dL_dfeature,
            # grads w.r. gaussian_idx_sorted,
            None,
            # grads w.r. tile_bins,
            None,
            # grads w.r. bg, 
            None,
            # grads w.r. W, 
            None,
            # grads w.r. H
            None
        )

        return grads
