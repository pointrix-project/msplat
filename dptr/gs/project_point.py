import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int
import dptr.gs._C as _C


def project_point(
    xyz: Float[Tensor, "P 3"],
    intrinsic: Float[Tensor, "4"],
    extrinsic: Float[Tensor, "3 4"],
    W: int,
    H: int,
    nearest: float = 0.2,
    extent: float = 1.3,
) -> Tuple:
    """
    Project 3D points to the screen.

    Parameters
    ----------
    xyz : Float[Tensor, "P 3"]
        3D position for each point.
    intrinsic : Float[Tensor, "4"]
        The intrinsic parameters of camera [fx, fy, cx, cy].
    extrinsic : Float[Tensor, "3 4"]
        The extrinsic parameters of camera [R|T].
    W : int
        Width of the image.
    H : int
        Height of the image.
    nearest : float, optional
        Nearest threshold for frustum culling, by default 0.2
    extent : float, optional
        Extent threshold for frustum culling, by default 1.3

    Returns
    -------
    uv : Float[Tensor, "P 2"]
        2D positions for each point in the image.
    depth: Float[Tensor, "P 1"]
        Depth for each point.
    """
    return _ProjectPoint.apply(xyz, intrinsic, extrinsic, W, H, nearest, extent)


class _ProjectPoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz: Float[Tensor, "*batch 3"],
        intrinsic: Float[Tensor, "4"],
        extrinsic: Float[Tensor, "3 4"],
        W: int,
        H: int,
        nearest: float,
        extent: float,
    ):
        (uv, depth) = _C.project_point_forward(
            xyz, intrinsic, extrinsic, W, H, nearest, extent
        )

        # save variables for backward
        ctx.W = W
        ctx.H = H
        ctx.save_for_backward(xyz, intrinsic, extrinsic, uv, depth)

        return uv, depth

    @staticmethod
    def backward(ctx, dL_duv, dL_ddepth):
        # get saved variables from forward
        H = ctx.H
        W = ctx.W
        xyz, intrinsic, extrinsic, uv, depth, uv, depth = ctx.saved_tensors

        (dL_dxyz, dL_dintrinsic, dL_dextrinsic) = _C.project_point_backward(
            xyz, intrinsic, extrinsic, W, H, uv, depth, dL_duv, dL_ddepth
        )

        grads = (
            # loss gradient w.r.t xyz
            dL_dxyz,
            # loss gradient w.r.t intrinsic
            dL_dintrinsic,
            # loss gradient w.r.t extrinsic
            dL_dextrinsic,
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
