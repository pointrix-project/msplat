import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int32
import msplat._C as _C


def sort_gaussian(
    uv: Float[Tensor, "P 2"],
    depth: Float[Tensor, "P 1"],
    W: int,
    H: int,
    radius: Int32[Tensor, "P 1"],
    tiles: Int32[Tensor, "P 1"],
) -> Tuple:
    """
    Sort 2D planar Gaussians base on [tile|depth].

    Parameters
    ----------
    uv : Float[Tensor, "P 2"]
        2D positions for each point in the image.
    depth : Float[Tensor, "P 1"]
        Depth for each point.
    W : int
        Width of the image.
    H : int
        Height of the image.
    radius : Int32[Tensor, "P 1"]
        Radius of 2D planar Gaussians on the image.
    tiles : Int32[Tensor, "P 1"]
        Number of tiles covered by 2D planar Gaussians on the image.

    Returns
    -------
    idx_sorted : Float[Tensor, "Nid"]
        Indices of Gaussian points sorted according to [tile_id|depth].
    tile_range :
        Range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
    """
    # accumulate intersections of gaussians
    tiles = torch.cumsum(tiles, dim=0, dtype=torch.int32)

    # compute gaussian keys for sorting
    (gaussian_key, gaussian_idx) = _C.compute_gaussian_key(
        uv, depth, W, H, radius, tiles
    )

    key_sorted, indices = torch.sort(gaussian_key)
    idx_sorted = torch.gather(gaussian_idx, 0, indices)

    tile_range = _C.compute_tile_gaussian_range(W, H, tiles, key_sorted)

    return idx_sorted, tile_range

def sort_gaussian2(
    uv,
    depth,
    W,
    H,
    radius,
    tiles,
) -> Tuple:
    """
    Sort 2D planar Gaussians base on [tile|depth].

    Parameters
    ----------
    uv : Float[Tensor, "P 2"]
        2D positions for each point in the image.
    depth : Float[Tensor, "P 1"]
        Depth for each point.
    W : int
        Width of the image.
    H : int
        Height of the image.
    radius : Int32[Tensor, "P 1"]
        Radius of 2D planar Gaussians on the image.
    tiles : Int32[Tensor, "P 1"]
        Number of tiles covered by 2D planar Gaussians on the image.

    Returns
    -------
    idx_sorted : Float[Tensor, "Nid"]
        Indices of Gaussian points sorted according to [tile_id|depth].
    tile_range :
        Range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
    """
    idx_sorted, tile_range = _C.sort_gaussian(uv, depth, W, H, radius, tiles)
    
    return idx_sorted, tile_range


def sort_gaussian_fast(
    uv,
    depth,
    W,
    H,
    radius,
    tiles,
) -> Tuple:
    """
    Sort 2D planar Gaussians base on [tile|depth].

    Parameters
    ----------
    uv : Float[Tensor, "P 2"]
        2D positions for each point in the image.
    depth : Float[Tensor, "P 1"]
        Depth for each point.
    W : int
        Width of the image.
    H : int
        Height of the image.
    radius : Int32[Tensor, "P 1"]
        Radius of 2D planar Gaussians on the image.
    tiles : Int32[Tensor, "P 1"]
        Number of tiles covered by 2D planar Gaussians on the image.

    Returns
    -------
    idx_sorted : Float[Tensor, "Nid"]
        Indices of Gaussian points sorted according to [tile_id|depth].
    tile_range :
        Range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
    """
    idx_sorted, tile_range = _C.sort_gaussian_fast(uv, depth, W, H, radius, tiles)
    
    return idx_sorted, tile_range
