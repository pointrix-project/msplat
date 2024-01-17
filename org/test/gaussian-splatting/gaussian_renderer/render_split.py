#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import sys

sys.path.append("/home/gj/code/DifferentiablePointRender/org")

from DifferentiablePointRender import (GaussianRasterizationSettings, compute_color_from_sh, compute_cov3d,
                                        gaussian_preprocess, gaussian_render)
    
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        debug=pipe.debug
    )
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width

    means3D = pc.get_xyz
    opacities = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation
    """
    p_hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1) @ viewpoint_camera.full_proj_transform
    p_proj = p_hom[:, :3] / (p_hom[:, 3:] + 0.0000001)
    p_view = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]
    depths = p_view[:, 2:3]
    means2D = torch.stack([
        ((p_proj[:, 0] + 1.0) * W - 1) * 0.5,
        ((p_proj[:, 1] + 1.0) * H - 1) * 0.5,
    ], dim=-1)
    
    visibility_filter = torch.logical_not((p_view[:, 2] <= 0.2) | (p_proj[:, 0] < -1.3) | (p_proj[:, 0] > 1.3) | (p_proj[:, 1] < -1.3) | (p_proj[:, 1] > 1.3))
    cov3D_precomp = compute_cov3d(scales, rotations, visibility_filter)
    # import pdb; pdb.set_trace()
    
    t = p_view
    t[:, :2] = torch.stack([
        (t[:, 0] / t[:, 2]).clamp(-1.3 * tanfovx, 1.3 * tanfovx) * t[:, 2],
        (t[:, 0] / t[:, 1]).clamp(-1.3 * tanfovy, 1.3 * tanfovy) * t[:, 2]
    ], dim=-1) 
    
    focal_y = H / (2 * tanfovy)
    focal_x = W / (2 * tanfovx)
    z = torch.zeros_like(t[:, 2])
    J = torch.stack([
        focal_x / t[:, 2], z, -(focal_x * t[:, 0]) / (t[:, 2] * t[:, 2]),
        z, focal_y / t[:, 2], -(focal_y * t[:, 1]) / (t[:, 2] * t[:, 2]),
        z, z, z
    ], dim=-1).reshape(-1, 3, 3)
    T = J @ viewpoint_camera.world_view_transform[:3, :3].T[None]
    Vrk = torch.stack([
        cov3D_precomp[:, 0], cov3D_precomp[:, 1], cov3D_precomp[:, 2],
        cov3D_precomp[:, 1], cov3D_precomp[:, 3], cov3D_precomp[:, 4],
        cov3D_precomp[:, 2], cov3D_precomp[:, 4], cov3D_precomp[:, 5],
    ], dim=-1).reshape(-1, 3, 3)
    # import pdb; pdb.set_trace()
    covs = T @ Vrk @ T.transpose(-1, -2)
    cov2Ds = torch.stack([
        covs[:, 0, 0] + 0.3, covs[:, 0, 1], covs[:, 1, 1] + 0.3
    ], dim=-1)
    
    det = cov2Ds[:, 0] * cov2Ds[:, 2] - cov2Ds[:, 1] * cov2Ds[:, 1]
    visibility_filter = torch.logical_and(visibility_filter, det != 0.0)
    det_inv = 1 / det

    mid = 0.5 * (cov2Ds[:, 0] + cov2Ds[:, 2]);
    lambda1 = mid + (mid * mid - det).clamp_min(0.1).sqrt()
    lambda2 = mid - (mid * mid - det).clamp_min(0.1).sqrt()
    radii = torch.ceil(3 * torch.maximum(lambda1, lambda2).sqrt()).int()
        
    BLOCK_X = BLOCK_Y = 16
    grid = [int((W + BLOCK_X - 1) / BLOCK_X), int((H + BLOCK_Y - 1) / BLOCK_Y)]
    rect_min = torch.stack([
        ((means2D[:, 0] - radii) / BLOCK_X).int().clamp(0, grid[0]),
        ((means2D[:, 1] - radii) / BLOCK_Y).int().clamp(0, grid[1]),
    ], dim=-1).int()
    rect_max = torch.stack([
        ((means2D[:, 0] + radii + BLOCK_X - 1) / BLOCK_X).int().clamp(0, grid[0]),
        ((means2D[:, 1] + radii + BLOCK_Y - 1) / BLOCK_Y).int().clamp(0, grid[1]),
    ], dim=-1).int()
    tiles_touched = (rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1])
    visibility_filter = torch.logical_and(visibility_filter, tiles_touched != 0)
    conics = torch.stack([
        cov2Ds[:, 0], -cov2Ds[:, 1], cov2Ds[:, 2]
    ], dim=-1) * det_inv[:, None]
    import pdb; pdb.set_trace()
    """
    visibility_filter = None
    cov3D_precomp = compute_cov3d(scales, rotations, visibility_filter)
    # scales must be empty if cov3D_precomp is not None
    scales = torch.Tensor([])
    rotations = torch.Tensor([])
    
    depths, radii, means2D, cov3Ds, conics, tiles_touched = gaussian_preprocess(
        means3D,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings,
    )
    means2D.retain_grad()
    
    # we can choose other mask
    visibility_filter = radii > 0
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    if pipe.convert_SHs_python:
        colors = torch.zeros(pc.get_xyz.size(0), 3, dtype=pc.get_xyz.dtype, device="cuda")
        shs_view = pc.get_features[visibility_filter].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized[visibility_filter])
        colors[visibility_filter] = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors = compute_color_from_sh(
            pc.active_sh_degree, 
            dir_pp_normalized, 
            pc.get_features,
            visibility_filter)
    
    features = colors
    
    feature = gaussian_render(
        features,
        depths, radii, means2D, conics, opacities, tiles_touched,
        visibility_filter,
        raster_settings,
    )
    
    rendered_image = feature[:3]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : visibility_filter,
            "radii": radii}
