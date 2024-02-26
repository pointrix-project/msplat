import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Bool
import dptr.gs._C as _C


def ewa_project(
    xyz: Float[Tensor, "P 3"],
    cov3d: Float[Tensor, "P 6"],
    intr: Float[Tensor, "4"],
    extr: Float[Tensor, "3 4"],
    uv: Float[Tensor, "P 4"],
    W: int,
    H: int,
    visible: Bool[Tensor, "P 1"] = None,
) -> Tuple:
    """
    Project 3D Gaussians to 2D planar Gaussian through elliptical weighted average(EWA).

    Parameters
    ----------
    xyz : Float[Tensor, "P 3"]
        3D position for each point.
    cov3d : Float[Tensor, "P 6"]
        The upper-right corner of the 3D covariance matrices, stored in a vector.
    intr : Float[Tensor, "4"]
        The intrinsic parameters of camera [fx, fy, cx, cy].
    extr : Float[Tensor, "3 4"]
        The extrinsic parameters of camera [R|T].
    uv : Float[Tensor, "P 4"]
        2D positions for each point in the image.
    W : int
        Width of the image.
    H : int
        Height of the image.
    visible : Bool[Tensor, "P 1"], optional
        The visibility status of each point, by default None

    Returns
    -------
    conic : Float[Tensor, "P 3"]
        The upper-right corner of the 2D covariance matrices, stored in a vector.
    radius : Int[Tensor, "P 1"]
        Radius of the 2D planar Gaussian on the image.
    tiles : Int[Tensor, "P 1"]
        Number of tiles covered by 2D planar Gaussians on the image.
    """

    if visible is None:
        visible = torch.ones_like(uv[:, 0], dtype=torch.bool)

    return _EWAProject.apply(xyz, cov3d, intr, extr, uv, W, H, visible)


class _EWAProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, cov3d, intr, extr, uv, W, H, visible):
        (conic, radius, tiles) = _C.ewa_project_forward(
            xyz, cov3d, intr, extr, uv, W, H, visible
        )

        ctx.save_for_backward(xyz, cov3d, intr, extr, radius)

        return conic, radius, tiles

    @staticmethod
    def backward(ctx, dL_dconic, dL_dradius, dL_dtiles):
        (xyz, cov3d, intr, extr, radius) = ctx.saved_tensors

        (dL_dxyz, dL_dcov3d, dL_dintr, dL_dextr) = _C.ewa_project_backward(
            xyz, cov3d, intr, extr, radius, dL_dconic
        )
        
        grads = (
            # loss gradient w.r.t xyz
            dL_dxyz,
            # loss gradient w.r.t cov3d
            dL_dcov3d,
            # loss gradient w.r.t intr
            dL_dintr if intr.requires_grad else None,
            # loss gradient w.r.t extr
            dL_dextr if extr.requires_grad else None,
            # loss gradient w.r.t uv
            None,
            # loss gradient w.r.t W,
            None,
            # loss gradient w.r.t H,
            None,
            # loss gradient w.r.t visible
            None,
        )

        return grads
