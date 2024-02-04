
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float
import DifferentiablePointRender.GaussianSplatting._C as _C

def sort_gaussian(
    uv: Float[Tensor, "num 2"],
    depth: Float[Tensor, "num 1"],
    w, h,
    radii: Float[Tensor, "num 1"],
    num_tiles_hit: Float[Tensor, "num 1"]
):
    # accumulate intersections of gaussians
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    
    # compute gaussian keys for sorting
    isect_ids, gaussian_ids = _C.compute_gaussian_key(
        uv,
        depth,
        w, h,
        radii,
        cum_tiles_hit
    )

    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
   
    tile_bins = _C.compute_tile_gaussian_range(
        w, h, 
        cum_tiles_hit,
        isect_ids_sorted)
    
    return gaussian_ids_sorted, tile_bins
