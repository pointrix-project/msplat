from torch import Tensor
from jaxtyping import Float
from .project_point import project_point
from .compute_cov3d import compute_cov3d
from .ewa_project import ewa_project
from .compute_sh import compute_sh
from .sort_gaussian import sort_gaussian
from .alpha_blending import alpha_blending


__all__ = [
    "project_point",
    "compute_cov3d",
    "ewa_project",
    "sort_gaussian",
    "compute_sh",
    "alpha_blending",
    "rasterization",
]


def rasterization(
    xyz: Float[Tensor, "P 3"],
    scale: Float[Tensor, "P 3"],
    rotate: Float[Tensor, "P 4"],
    opacity: Float[Tensor, "P 1"],
    feature: Float[Tensor, "P C"],
    viewmat: Float[Tensor, "4 4"],
    projmat: Float[Tensor, "4 4"],
    camparam: Float[Tensor, "4"],
    W: int,
    H: int,
    bg: float,
) -> Float[Tensor, "C H W"]:
    """
    Vanilla 3D Gaussian Splatting rasterization pipeline.

    Parameters
    ----------
    xyz : Float[Tensor, "P 3"]
        3D position for each point.
    scale : Float[Tensor, "P 3"]
        3D scaleing vector for each point.
    rotate : Float[Tensor, "P 4"]
        3D rotations (unit quaternions) for each point.
    opacity : Float[Tensor, "P 1"]
        Opacity values for each point.
    feature : Float[Tensor, "P C"]
        Features for each point to be alpha blended.
    viewmat : Float[Tensor, "4 4"]
        The world to view transform matrix.
    projmat : Float[Tensor, "4 4"]
        The world to screen transform matrix.
    camparam : Float[Tensor, "4"]
        The intrinsics of camera [fx, fy, cx, cy].
    W : int
        Width of the image.
    H : int
        Height of the image.
    bg : float
        Background color.

    Returns
    -------
    feature_map : Float[Tensor, "C H W"]
        Rendered feature maps.
    """
    # project points
    (uv, depth) = project_point(xyz, viewmat, projmat, camparam, W, H)

    visible = depth != 0

    # compute cov3d
    cov3d = compute_cov3d(scale, rotate, visible)

    # ewa project
    (conic, radius, tiles_touched) = ewa_project(
        xyz, cov3d, viewmat, camparam, uv, W, H, visible
    )

    # sort
    (gaussian_ids_sorted, tile_range) = sort_gaussian(
        uv, depth, W, H, radius, tiles_touched
    )

    # alpha blending
    render_feature = alpha_blending(
        uv, conic, opacity, feature, gaussian_ids_sorted, tile_range, bg, W, H
    )

    return render_feature
