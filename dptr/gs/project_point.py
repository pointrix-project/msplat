
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int
import dptr.gs._C as _C


def project_point(
    xyz: Float[Tensor, "*batch 3"],
    viewmat: Float[Tensor, "4 4"],
    projmat: Float[Tensor, "4 4"],
    camparam: Float[Tensor, "4"],
    W: int, H: int,
    nearest: float = 0.2,
    extent: float = 1.3
)->Tuple[Tensor, Tensor]:
    """_summary_
    :param xyz: _description_
    :type xyz: Float[Tensor, 
    :param viewmat: _description_
    :type viewmat: Float[Tensor, &quot;4 4&quot;]
    :param projmat: _description_
    :type projmat: Float[Tensor, &quot;4 4&quot;]
    :param camparam: _description_
    :type camparam: Float[Tensor, &quot;4&quot;]
    :param W: _description_
    :type W: int
    :param H: _description_
    :type H: int
    :param nearest: _description_, defaults to 0.2
    :type nearest: float, optional
    :param extent: _description_, defaults to -1
    :type extent: float, optional
    :return: _description_
    :rtype: Tuple[Tensor, Tensor]
    """
    return _ProjectPoint.apply(
        xyz,
        viewmat,
        projmat,
        camparam,
        W, H,
        nearest,
        extent
    )


class _ProjectPoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz: Float[Tensor, "*batch 3"],
        viewmat: Float[Tensor, "4 4"],
        projmat: Float[Tensor, "4 4"],
        camparam: Float[Tensor, "4"],
        W: int, H: int,
        nearest: float,
        extent: float
    ):
        
        (
            uv, 
            depth
        ) = _C.project_point_forward(
            xyz, 
            viewmat, 
            projmat,
            camparam,
            W, H,
            nearest,
            extent
        )
        
        # save variables for backward
        ctx.W = W
        ctx.H = H
        ctx.save_for_backward(
            xyz, 
            viewmat, 
            projmat,
            camparam,
            uv,
            depth
            )
        
        return uv, depth

    @staticmethod
    def backward(ctx, dL_duv, dL_ddepth):
        
        # get saved variables from forward
        H = ctx.H
        W = ctx.W
        xyz, viewmat, projmat, camparam, uv, depth = ctx.saved_tensors
        
        dL_dxyz = _C.project_point_backward(
            xyz, 
            viewmat,
            projmat,
            camparam,
            W, H,
            uv,
            depth, 
            dL_duv,
            dL_ddepth
        )
        
        grads = (
            # loss gradient w.r.t xyz
            dL_dxyz,
            # loss gradient w.r.t viewmat
            None,
            # loss gradient w.r.t projmat
            None,
            # loss gradient w.r.t camparam
            None,
            # loss gradient w.r.t W
            None,
            # loss gradient w.r.t H
            None,
            # loss gradient w.r.t nearest
            None,
            # loss gradient w.r.t extent
            None,
        )

        return grads
