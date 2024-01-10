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
from DifferentiablePointRender import (GaussianRasterizationSettings, compute_color_from_sh,
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

    means3D = pc.get_xyz
    opacities = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    cov3D_precomp = torch.Tensor([])
    
    # with torch.no_grad():
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
