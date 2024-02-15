
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

# GS_Split
# from typing import NamedTuple
# from GS_Split import _C as gs_split_c


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
            "scale": torch.exp,
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


# # GS_Split
# class _GaussianRender(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         features,
#         depths,  # used for sorting 
#         radii, 
#         means2D, 
#         conics, 
#         opacities, 
#         tiles_touched,
#         visibility_filter,
#         raster_settings,
#     ):
#         if visibility_filter is None:
#             visibility_filter = torch.ones_like(features[:, 0], dtype=torch.bool)
#         # Restructure arguments the way that the C++ lib expects them
#         args = (
#             features,
#             depths, radii, means2D, conics, opacities, tiles_touched,
#             visibility_filter,
#             raster_settings.image_height,
#             raster_settings.image_width,
#             raster_settings.debug
#         )
#         num_rendered, feature, binningBuffer, imgBuffer = gs_split_c.render_forward(*args)
        
#         # Keep relevant tensors for backward
#         ctx.raster_settings = raster_settings
#         ctx.num_rendered = num_rendered
#         ctx.save_for_backward(features, means2D, conics, opacities, binningBuffer, imgBuffer)
#         return feature

#     @staticmethod
#     def backward(ctx, dL_dfeatures):

#         # Restore necessary values from context
#         num_rendered = ctx.num_rendered
#         raster_settings = ctx.raster_settings
#         features, means2D, conics, opacities, binningBuffer, imgBuffer = ctx.saved_tensors

#         # Restructure args as C++ method expects them
#         args = (features, means2D, conics, opacities,
#                 dL_dfeatures, 
#                 num_rendered,
#                 binningBuffer,
#                 imgBuffer,
#                 raster_settings.debug)

#         dL_dfeatures, dL_dmeans2D, dL_dcov3Ds, dL_dconics = gs_split_c.render_backward(*args)

#         grads = (
#             dL_dfeatures,
#             None,  # depth
#             None,  # grad_radii,
#             dL_dmeans2D,
#             dL_dcov3Ds,
#             dL_dconics,
#             None,  # grad_tiles_touched,
#             None,  # grad_visibility_filter,
#             None,  # raster_settings
#         )
#         return grads
    
# gaussian_render = _GaussianRender.apply

# class GaussianRasterizationSettings(NamedTuple):
#     image_height: int
#     image_width: int 
#     tanfovx : float
#     tanfovy : float
#     viewmatrix : torch.Tensor
#     projmatrix : torch.Tensor
#     debug : bool
    

# class _ComputeCov3D(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         scales,
#         rotations,
#         visibility_filter=None
#     ):
#         if visibility_filter is None:
#             visibility_filter = torch.ones_like(scales[:, 0], dtype=torch.bool)
#         # Restructure arguments the way that the C++ lib expects them
#         args = (
#             scales,
#             rotations,
#             visibility_filter,
#         )
#         cov3Ds = gs_split_c.compute_cov3d_forward(*args)
#         ctx.save_for_backward(scales, rotations, visibility_filter)
#         return cov3Ds

#     @staticmethod
#     def backward(ctx, dL_dcov3Ds):
#         scales, rotations, visibility_filter = ctx.saved_tensors

#         # Restructure args as C++ method expects them
#         args = (
#             scales,
#             rotations,
#             visibility_filter,
#             dL_dcov3Ds,
#         )

#         dL_dscales, dL_drotations = gs_split_c.compute_cov3d_backward(*args)

#         grads = (
#             dL_dscales,
#             dL_drotations,
#             None,  # visibility_filter
#         )

#         return grads

# compute_cov3d = _ComputeCov3D.apply


# class _GaussianPreprocess(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         means3D,
#         scales,
#         rotations,
#         cov3Ds_precomp,
#         raster_settings,
#     ):
#         # Restructure arguments the way that the C++ lib expects them
#         args = (
#             means3D,
#             scales,
#             rotations,
#             cov3Ds_precomp,
#             raster_settings.viewmatrix,
#             raster_settings.projmatrix,
#             raster_settings.tanfovx,
#             raster_settings.tanfovy,
#             raster_settings.image_height,
#             raster_settings.image_width,
#             raster_settings.debug
#         )
#         depths, radii, means2D, cov3Ds, conics, tiles_touched = gs_split_c.preprocess_forward(*args)
#         # Keep relevant tensors for backward
#         ctx.raster_settings = raster_settings
#         ctx.save_for_backward(means3D, scales, rotations, cov3Ds_precomp, depths, radii, means2D, cov3Ds, conics, tiles_touched)
#         return depths, radii, means2D, cov3Ds, conics, tiles_touched

#     @staticmethod
#     def backward(ctx, dL_ddepths, dL_dradii, dL_dmeans2D, dL_dcov3Ds, dL_dconics, dL_dtiles_touched):

#         # Restore necessary values from context
#         raster_settings = ctx.raster_settings
#         means3D, scales, rotations, cov3Ds_precomp, depths, radii, means2D, cov3Ds, conics, tiles_touched = ctx.saved_tensors
        
#         # Restructure args as C++ method expects them
#         args = (means3D, 
#                 radii, 
#                 scales, 
#                 rotations, 
#                 cov3Ds, 
#                 cov3Ds_precomp, 
#                 raster_settings.viewmatrix, 
#                 raster_settings.projmatrix, 
#                 raster_settings.tanfovx, 
#                 raster_settings.tanfovy, 
#                 raster_settings.image_height,
#                 raster_settings.image_width,
#                 dL_ddepths,
#                 dL_dmeans2D,
#                 dL_dcov3Ds,
#                 dL_dconics,
#                 raster_settings.debug)

#         dL_dmeans3D, dL_dscales, dL_drotations, dL_dcov3Ds_precomp = gs_split_c.preprocess_backward(*args)

#         grads = (
#             dL_dmeans3D,
#             dL_dscales,
#             dL_drotations,
#             dL_dcov3Ds_precomp,
#             None,  # raster_settings
#         )
#         return grads

# gaussian_preprocess = _GaussianPreprocess.apply

  
if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    
    bg = 0
    image_file = "./media/DPTR.png"
    img = np.array(Image.open(image_file))
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1)
    
    C, H, W = gt.shape 
    
    bg = 0
    fov_x = math.pi / 2.0
    fx = 0.5 * float(W) / math.tan(0.5 * fov_x)
    camparam = torch.Tensor([fx, fx, float(W) / 2, float(H) / 2]).cuda().float()
    viewmat = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 8.0, 1.0]]).cuda().float()
    projmat = viewmat.clone()
    
    gaussians = SimpleGaussian(num_points=100000)
    
    max_iter = 7000
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    l1_loss = nn.L1Loss()
    
    # GS_Split
    # setting = GaussianRasterizationSettings
    # setting.image_height = H
    # setting.image_width = W
    # setting.tanfovx = math.tan(0.5 * fov_x)
    # setting.tanfovy = math.tan(0.5 * fov_x)
    # setting.viewmatrix = viewmat
    # setting.projmatrix = projmat
    # setting.debug = False
    
    for iteration in range(0, max_iter):
        
        # # print("GS_cov3d")
        # cov3d = compute_cov3d(
        #     gaussians.get_attribute("scale"), 
        #     gaussians.get_attribute("rotate")
        # )
        
        # # print("Preprocess GS")
        # (
        #     depth, 
        #     radius, 
        #     uv, 
        #     cov3d,
        #     conic,
        #     tiles_touched
        # ) = gaussian_preprocess(
        #     gaussians.get_attribute("xyz"), 
        #     gaussians.get_attribute("scale"), 
        #     gaussians.get_attribute("rotate"), 
        #     cov3d, 
        #     setting
        # )
        
        # # print("Render GS")
        # render_feature = gaussian_render(
        #     gaussians.get_attribute("rgb"),
        #     depth,
        #     radius,
        #     uv,
        #     conic,
        #     gaussians.get_attribute("opacity"),
        #     tiles_touched,
        #     None,
        #     setting
        # )
        
        # project points
        (
            uv,
            depth 
        ) = gs.project_point(
            gaussians.get_attribute("xyz"), 
            viewmat, 
            projmat,
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
        
        # valid_num = torch.sum(visibility_status.float())
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 100 == 0:
            show_data = render_feature.detach().permute(1, 2, 0)
            show_data = torch.clamp(show_data, 0.0, 1.0)
            frames.append((show_data.cpu().numpy() * 255).astype(np.uint8))
    
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