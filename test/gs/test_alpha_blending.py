
import math
import time
import torch
import numpy as np
import DifferentiablePointRender.GaussianSplatting as gs
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

    return out_img.permute(2, 0, 1), final_Ts, final_idx

if __name__ == "__main__":

    seed = 123
    torch.manual_seed(seed)
    
    w = 4
    h = 5
    bg = 0
    
    x_coords = torch.arange(w).cuda()
    y_coords = torch.arange(h).cuda()
    xx, yy = torch.meshgrid(x_coords, y_coords)
    uv = torch.stack((xx.flatten(), yy.flatten()), dim=1).cuda().float()
    conic = torch.ones([h*w, 3], device="cuda", dtype=torch.float32)
    conic[:, 1] = 0
    
    depth = torch.rand_like(uv[:, 0:1]) * 5
    radii = torch.ones_like(depth).int()
    num_tiles_hit = torch.ones_like(radii)
    opacity = torch.ones_like(depth)
    feature = torch.ones([h*w, 3], device="cuda", dtype=torch.float32) + 1
    
    for i in range(feature.shape[-1]):
        feature[:, i] = i + 1

    gaussian_ids_sorted, tile_bins = gs.sort_gaussian(uv, depth, w, h, radii, num_tiles_hit)

    render_feature_cuda, final_T_cuda, ncontrib_cuda = gs.alpha_blending(
        uv, conic, opacity, feature, 
        gaussian_ids_sorted, tile_bins, bg, w, h)
    
    render_feature_torch, final_T_torch, ncontrib_torch = alpha_blending_torch_impl(
        uv,
        conic,
        opacity,
        feature,
        gaussian_ids_sorted,
        tile_bins,
        bg,
        w, 
        h
    )
    
    torch.testing.assert_close(render_feature_torch, render_feature_cuda)
    torch.testing.assert_close(final_T_torch, final_T_cuda)
    torch.testing.assert_close(ncontrib_torch, ncontrib_cuda)
    
    # grid = int(np.ceil(np.sqrt(render_feature_cuda.shape[0])))

    # for i in range(grid * grid):
    #     if i >= render_feature_cuda.shape[0]:
    #         break
    #     plt.subplot(grid, grid, i+1)
    #     plt.imshow(render_feature_cuda[i].cpu().numpy())
    # plt.savefig("render_feature_cuda.png")
    
    # plt.clf()
    # plt.imshow(final_T_cuda.cpu().numpy())
    # plt.savefig("final_T_cuda.png")
    
    # plt.clf()
    # plt.imshow(ncontrib_cuda.cpu().numpy())
    # plt.savefig("ncontrib_cuda.png")
    
    # plt.clf()
    # for i in range(grid * grid):
    #     if i >= render_feature_torch.shape[0]:
    #         break
    #     plt.subplot(grid, grid, i+1)
    #     plt.imshow(render_feature_torch[i].cpu().numpy())
    # plt.savefig("render_feature_torch.png")
    
    # plt.clf()
    # plt.imshow(final_T_torch.cpu().numpy())
    # plt.savefig("final_T_torch.png")
    
    # plt.clf()
    # plt.imshow(ncontrib_torch.cpu().numpy())
    # plt.savefig("ncontrib_torch.png")
    