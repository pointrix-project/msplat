
import math
import time
import torch
import numpy as np
import msplat as ms
import matplotlib.pyplot as plt

def get_tiles(uv, radius, W, H):
    BLOCK_X = 16
    BLOCK_Y = 16
    
    # get tiles
    top_left = torch.zeros_like(uv, dtype=torch.int, device=uv.device)
    bottom_right = torch.zeros_like(uv, dtype=torch.int, device=uv.device)
    
    top_left[:, 0] = ((uv[:, 0] - radius) / BLOCK_X)
    top_left[:, 1] = ((uv[:, 1] - radius) / BLOCK_Y)
    bottom_right[:, 0] = ((uv[:, 0] + radius + BLOCK_X - 1) / BLOCK_X)
    bottom_right[:, 1] = ((uv[:, 1] + radius + BLOCK_Y - 1) / BLOCK_Y)
    
    tile_bounds = torch.zeros(2, dtype=torch.int, device=uv.device)
    tile_bounds[0] = (W + BLOCK_X - 1) / BLOCK_X
    tile_bounds[1] = (H + BLOCK_Y - 1) / BLOCK_Y
    
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    
    tiles_tmp = tile_max - tile_min
    tiles = tiles_tmp[..., 0] * tiles_tmp[..., 1]
    
    return tiles

def generate_covariance2d():
    A_batch = torch.randn(N, 2, 2).cuda()
    cov = torch.bmm(A_batch, A_batch.transpose(1, 2))
    
    return torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]], dim=-1)

if __name__ == "__main__":
    
    for c in range(1, 42):
        iters = 100
        
        seed = 121
        torch.manual_seed(seed)
        
        print("=============================== running test on alpha_blending ===============================")
        
        # generate data
        w = 800
        h = 800
        bg = 0
        
        N = 10000
        
        uv = torch.rand([N, 2], device="cuda", dtype=torch.float32)
        uv[:, 0] = uv[:, 0] * w
        uv[:, 1] = uv[:, 1] * h
        
        conic = generate_covariance2d()
        
        depth = torch.rand_like(uv[:, 0:1]) * 5
        radius = (torch.rand_like(depth) * 5).int()
        tiles = get_tiles(uv, radius.squeeze(-1), w, h)
        opacity = torch.rand_like(depth)
        feature = torch.rand([N, c], device="cuda", dtype=torch.float32)
        
        # sort
        (
            idx_sorted, 
            tile_range
        ) = ms.sort_gaussian(
            uv, 
            depth, 
            w, 
            h, 
            radius, 
            tiles
        )
        
        uv = uv.requires_grad_()
        conic = conic.requires_grad_()
        opacity = opacity.requires_grad_()
        feature = feature.requires_grad_()
        
        # ============================================ Forward =====================================
        t = time.time()
        for i in range(iters):
            rendered_feature = ms.alpha_blending(
                uv, 
                conic, 
                opacity, 
                feature, 
                idx_sorted, 
                tile_range, 
                bg, 
                w, 
                h
            )
            loss = rendered_feature.sum()
            loss.backward()
            
        torch.cuda.synchronize()
        
        cost = (time.time() - t) / iters
        print("channel: ", c, " cuda runtime: ", cost, " s")
    