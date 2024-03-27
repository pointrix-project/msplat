
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat

class SimpleGaussian:
    def __init__(self, num_points=100000):
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   torch.rand((N, 4), dtype=torch.float32).cuda(),
            "opacity":  torch.rand((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }
        
        self._activations = {
            "scale": lambda x: torch.abs(x) + 1e-8,
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.sigmoid,
            "rgb": torch.sigmoid
        }
        
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=0.01)
        
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    
    img = imageio.imread("./data/stanford-bunny.jpg")
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1)
    
    C, H, W = gt.shape 
    
    bg = 1
    fov = math.pi / 2.0
    fx = 0.5 * float(W) / math.tan(0.5 * fov)
    fy = 0.5 * float(H) / math.tan(0.5 * fov)
    intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float()
    extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 2.5]]).cuda().float()
    
    gaussians = SimpleGaussian(num_points=10000)
    
    max_iter = 7000
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    cal_loss = nn.SmoothL1Loss()
    
    for iteration in range(0, max_iter):
        
        rendered_feature = msplat.rasterization(
            gaussians.get_attribute("xyz"),
            gaussians.get_attribute("scale"),
            gaussians.get_attribute("rotate"), 
            gaussians.get_attribute("opacity"),
            gaussians.get_attribute("rgb"),
            intr,
            extr,
            W, H, bg)
        
        loss = cal_loss(rendered_feature, gt)
        loss.backward()
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 20 == 0:
            render = rendered_feature.detach().permute(1, 2, 0)
            render = torch.clamp(render, 0.0, 1.0)
            render = (render.cpu().numpy() * 255).astype(np.uint8)
            
            empty = np.ones((render.shape[0], 2, 3), dtype=np.uint8)
            show_data = np.hstack((render, empty, (img * 255).astype(np.uint8)))
            frames.append(show_data)
    
    progress_bar.close()
    
    # save them as a video with imageio
    frames = np.stack(frames, axis=0)
    imageio.mimwrite("tutorial_2d.mp4", frames, fps=30)
    