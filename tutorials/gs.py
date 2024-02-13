
# This is a script for Tutorial on Differentiable Point Renderer.
# The task finished here is to optimize 3D Gaussian for a single image based on dptr.gs

import os
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
import dptr.gs as gs
import matplotlib.pyplot as plt
from PIL import Image


def generate_random_quats(N):
    u = torch.rand((N, 1)).cuda()
    v = torch.rand((N, 1)).cuda()
    w = torch.rand((N, 1)).cuda()
    quats = torch.cat([torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                        torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                        torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                        torch.sqrt(u) * torch.cos(2.0 * math.pi * w)], dim=-1)
    return quats

class SimpleGaussian:
    def __init__(self, num_points=1e+5):
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   generate_random_quats(N),
            "opacity":  torch.ones((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }
        
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
            
        params = [
            {'params': [self._attributes["xyz"]], "name": "xyz"},
            {'params': [self._attributes["scale"]], "name": "scaling"},
            {'params': [self._attributes["rotate"]], "name": "rotation"},
            {'params': [self._attributes["opacity"]], "name": "opacity"},
            {'params': [self._attributes["rgb"]], "name": "rgb"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.01)
        
    def step(self):
        if self.optimizer is None:
            raise ValueError("The optimizer should be set.")
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_attribute(self, name):
        try:
            return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

    
if __name__ == "__main__":
    seed = 121
    torch.manual_seed(seed)
    
    max_iter = 1000
    bg = 0
    fov_x = math.pi / 2.0
    
    image_file = "./media/DPTR.png"
    img = np.array(Image.open(image_file))
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1)
    
    C, H, W = gt.shape    
    fx = 0.5 * float(W) / math.tan(0.5 * fov_x)
    camparam = torch.tensor([fx, fx, 0, 0]).cuda().float()
    viewmat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 8.0],
                            [0.0, 0.0, 0.0, 1.0],], dtype=torch.float32).cuda()
    
    gaussians = SimpleGaussian(num_points=1000)
    
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    l1_loss = nn.L1Loss()
    for iteration in range(1, max_iter+1):
        # project points
        (
            uv,
            depth 
        ) = gs.project_point(
            gaussians.get_attribute("xyz"), 
            viewmat, 
            viewmat,
            camparam,
            W, H)
        
        visibility_status = depth != 0
        
        # compute cov3d
        cov3d = gs.compute_cov3d(
            gaussians.get_attribute("scale"), 
            gaussians.get_attribute("rotate"), 
            visibility_status)
        
        # ewa project
        (
            conic, 
            radius, 
            tiles_touched
        ) = gs.ewa_project(
            gaussians.get_attribute("xyz"),
            cov3d, 
            viewmat,
            camparam,
            uv,
            W, H,
            visibility_status
        )
        
        # sort
        (
            gaussian_ids_sorted, 
            tile_bins
        ) = gs.sort_gaussian(
            uv, 
            depth, 
            W, H, 
            radius, 
            tiles_touched
        )
        
        # alpha blending
        (
            render_feature, 
            final_T, 
            ncontrib
        ) = gs.alpha_blending(
            uv, 
            conic, 
            gaussians.get_attribute("opacity"), 
            gaussians.get_attribute("rgb"), 
            gaussian_ids_sorted, 
            tile_bins, 
            bg, 
            W, 
            H
        )
        
        loss = l1_loss(render_feature, gt)
        loss.backward()
        
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 50 == 0:
            frames.append((render_feature.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    
    progress_bar.close()
    
    # save them as a gif with PIL
    frames = [Image.fromarray(frame) for frame in frames]
    out_dir = "media"
    os.makedirs(out_dir, exist_ok=True)
    frames[0].save(
        f"{out_dir}/tutorial.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,
    )