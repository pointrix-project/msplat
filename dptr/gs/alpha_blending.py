import torch
from torch import Tensor
from jaxtyping import Float
import dptr.gs._C as _C


def alpha_blending(
    uv: Float[Tensor, "P 2"],
    conic: Float[Tensor, "P 2"],
    opacity: Float[Tensor, "P 1"],
    feature: Float[Tensor, "P C"],
    idx_sorted: Float[Tensor, "Nid"],
    title_bins: Float[Tensor, "Ntile 2"],
    bg: float,
    W: int,
    H: int,
    ndc: Float[Tensor, "P 2"]=None
) -> Float[Tensor, "C H W"]:
    """
    Alpha Blending for sorted 2D planar Gaussian in a tile based manner.

    Parameters
    ----------
    uv : Float[Tensor, "P 2"]
        2D positions for each point in the image.
    conic : Float[Tensor, "P 2"]
        Inverse 2D covariances for each point in the image.
    opacity : Float[Tensor, "P 1"]
        Opacity values for each point.
    feature : Float[Tensor, "P C"]
        Features for each point to be alpha blended.
    idx_sorted : Float[Tensor, "Nid"]
        Indices of Gaussian points sorted according to [tile_id|depth].
    title_bins : Float[Tensor, "Ntile 2"]
        Range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
    bg : float
        Background color.
    W : int
        Width of the image.
    H : int
        Height of the image.
    ndc: Float[Tensor, "P 2"]
        Just for storing the gradients of NDC coordinates for adaptive density control, by default None

    Returns
    -------
    feature_map : Float[Tensor, "C H W"]
        Rendered feature maps.
    """

    return _AlphaBlending.apply(
        uv, conic, opacity, feature, idx_sorted, title_bins, bg, W, H, ndc
    )


class _AlphaBlending(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv, conic, opacity, feature, idx_sorted, tile_range, bg, W, H, ndc):
        (render_feature, final_T, ncontrib) = _C.alpha_blending_forward(
            uv, conic, opacity, feature, idx_sorted, tile_range, bg, W, H
        )

        ctx.W = W
        ctx.H = H
        ctx.bg = bg
        ctx.ndc = ndc
        
        ctx.save_for_backward(
            uv, conic, opacity, feature, idx_sorted, tile_range, final_T, ncontrib
        )

        return render_feature

    @staticmethod
    def backward(ctx, dL_drendered):
        W = ctx.W
        H = ctx.H
        bg = ctx.bg
        ndc = ctx.ndc

        (
            uv,
            conic,
            opacity,
            feature,
            idx_sorted,
            tile_range,
            final_T,
            ncontrib,
        ) = ctx.saved_tensors

        (dL_duv, dL_dconic, dL_dopacity, dL_dfeature) = _C.alpha_blending_backward(
            uv,
            conic,
            opacity,
            feature,
            idx_sorted,
            tile_range,
            bg,
            W,
            H,
            final_T,
            ncontrib,
            dL_drendered,
        )
        
        dL_dndc = None
        if ndc is not None:
            duv_dndc = torch.tensor([0.5 * W, 0.5 * H], dtype=uv.dtype, device=uv.device)
            dL_dndc = dL_duv * duv_dndc[None, ...]

        grads = (
            # grads w.r.t uv
            dL_duv,
            # grads w.r. conic,
            dL_dconic,
            # grads w.r. opacity,
            dL_dopacity,
            # grads w.r. feature,
            dL_dfeature,
            # grads w.r. idx_sorted,
            None,
            # grads w.r. tile_range,
            None,
            # grads w.r. bg,
            None,
            # grads w.r. W,
            None,
            # grads w.r. H
            None,
            # grads w.r. ndc
            dL_dndc
        )

        return grads
