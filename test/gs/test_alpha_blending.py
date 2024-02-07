
import time
import torch
import numpy as np
import DifferentiablePointRender.GaussianSplatting as gs
import matplotlib.pyplot as plt

if __name__ == "__main__":

    seed = 123
    torch.manual_seed(seed)
    
    # feature = torch.rand([2, 3, 3], device="cuda", dtype=torch.float32)
    
    # # for i in range(feature.shape[-1]):
    # #     feature[:, :, i] = i + 1
    
    # print(feature.reshape(-1))
    # exit(-1)

    w = 160
    h = 160
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
    feature = torch.ones([w*h, 33], device="cuda", dtype=torch.float32) + 1
    
    for i in range(feature.shape[-1]):
        feature[:, i] = i + 1

    gaussian_ids_sorted, tile_bins = gs.sort_gaussian(uv, depth, w, h, radii, num_tiles_hit)

    render_feature = gs.alpha_blending(
        uv, conic, opacity, feature, 
        gaussian_ids_sorted, tile_bins, bg, w, h)
    print(render_feature.shape)
    
    grid = int(np.ceil(np.sqrt(render_feature.shape[0])))

    for i in range(grid * grid):
        if i >= render_feature.shape[0]:
            break
        plt.subplot(grid, grid, i+1)
        plt.imshow(render_feature[i].cpu().numpy())
    
    plt.show()
