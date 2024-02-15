
import math
import time
import torch
import numpy as np
import dptr.gs as gs
import matplotlib.pyplot as plt

def alpha_blending_torch_impl(
    uv,
    conic,
    opacity,
    feature,
    gaussian_ids_sorted,
    tile_bins,
    bg,
    W, H
):
    channels = feature.shape[1]
    out_img = torch.zeros((H, W, channels), dtype=torch.float32, device=uv.device)
    final_Ts = torch.zeros((H, W), dtype=torch.float32, device=uv.device)
    final_idx = torch.zeros((H, W), dtype=torch.int32, device=uv.device)
    
    for i in range(H):
        for j in range(W):
            tile_id = (i // 16) * math.ceil(W / 16) + (j // 16)
            tile_bin_start = tile_bins[tile_id, 0]
            tile_bin_end = tile_bins[tile_id, 1]
            
            T = 1.0
            last_contributor = 0
            contributor = 0
            for idx in range(tile_bin_start, tile_bin_end):
                contributor += 1
                
                gaussian_id = gaussian_ids_sorted[idx]
                conic_idx = conic[gaussian_id]
                center = uv[gaussian_id]
                delta = center - torch.tensor(
                    [j, i], dtype=torch.float32, device=uv.device
                )
                
                power = -(
                    0.5 * conic_idx[0] * delta[0] * delta[0] 
                    + 0.5 * conic_idx[2] * delta[1] * delta[1]
                    + conic_idx[1] * delta[0] * delta[1])

                opac = opacity[gaussian_id]
                alpha = min(0.99, opac * torch.exp(power))
                
                if power > 0 or alpha < 1.0 / 255.0:
                    continue

                next_T = T * (1 - alpha)
                if next_T <= 1e-4:
                    break
                
                out_img[i, j] += alpha * T * feature[gaussian_id]
                T = next_T
                last_contributor = contributor

            final_Ts[i, j] = T
            final_idx[i, j] = last_contributor
            out_img[i, j] += T * bg

    return out_img.permute(2, 0, 1)


def get_touched_tiles(uv, radius, W, H):
    BLOCK_X = 16
    BLOCK_Y = 16
    
    # get tiles_touched
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
    tiles_touched = tiles_tmp[..., 0] * tiles_tmp[..., 1]
    
    return tiles_touched


def generate_covariance2d():
    A_batch = torch.randn(N, 2, 2).cuda()
    cov = torch.bmm(A_batch, A_batch.transpose(1, 2))
    
    return torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]], dim=-1)

if __name__ == "__main__":

    # seed = 121
    # torch.manual_seed(seed)
    
    print("=============================== running test on alpha_blending ===============================")
    
    # generate data
    w = 3
    h = 3
    bg = 1
    
    N = 20
    
    uv = torch.rand([N, 2], device="cuda", dtype=torch.float32)
    uv[:, 0] = uv[:, 0] * w
    uv[:, 1] = uv[:, 1] * h
    
    conic = generate_covariance2d()
    
    depth = torch.rand_like(uv[:, 0:1]) * 5
    radii = (torch.rand_like(depth) * 5).int()
    num_tiles_hit = get_touched_tiles(uv, radii.squeeze(-1), w, h)
    opacity = torch.rand_like(depth)
    feature = torch.rand([N, 2], device="cuda", dtype=torch.float32)
    
    # sort
    (
        gaussian_ids_sorted, 
        tile_bins
    ) = gs.sort_gaussian(
        uv, 
        depth, 
        w, 
        h, 
        radii, 
        num_tiles_hit
    )
    
    # ============================================ Forward =====================================
    print("forward: ")
    uv1 = uv.clone().requires_grad_()
    uv2 = uv.clone().requires_grad_()
    conic1 = conic.clone().requires_grad_()
    conic2 = conic.clone().requires_grad_()
    opacity1 = opacity.clone().requires_grad_()
    opacity2 = opacity.clone().requires_grad_()
    feature1 = feature.clone().requires_grad_()
    feature2 = feature.clone().requires_grad_()
    
    render_feature_torch = alpha_blending_torch_impl(
        uv1,
        conic1,
        opacity1,
        feature1,
        gaussian_ids_sorted,
        tile_bins,
        bg,
        w, 
        h
    )
    
    render_feature_cuda = gs.alpha_blending(
        uv2, 
        conic2, 
        opacity2, 
        feature2, 
        gaussian_ids_sorted, 
        tile_bins, 
        bg, 
        w, 
        h
    )
    
    torch.testing.assert_close(render_feature_torch, render_feature_cuda)
    print("Forward pass.")
    
    print("Backward: ")
    loss1 = render_feature_torch.sum()
    loss1.backward()
    
    loss2 = render_feature_cuda.sum()
    loss2.backward()
    
    torch.testing.assert_close(uv1.grad, uv2.grad)
    torch.testing.assert_close(conic1.grad, conic2.grad)
    torch.testing.assert_close(opacity1.grad, opacity2.grad)
    torch.testing.assert_close(feature1.grad, feature2.grad)
    print("Backward pass.")
    