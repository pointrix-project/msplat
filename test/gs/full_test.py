
import torch
import math
import sys
import os

# os.system("pip install .")
# os.system("pip install ./GS_Split")

import DifferentiablePointRender.GaussianSplatting as gs

from typing import NamedTuple
from torch.utils.cpp_extension import load
from GS_Split import _C

class _ComputeCov3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scales,
        rotations,
        visibility_filter=None
    ):
        if visibility_filter is None:
            visibility_filter = torch.ones_like(scales[:, 0], dtype=torch.bool)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            scales,
            rotations,
            visibility_filter,
        )
        cov3Ds = _C.compute_cov3d_forward(*args)
        ctx.save_for_backward(scales, rotations, visibility_filter)
        return cov3Ds

    @staticmethod
    def backward(ctx, dL_dcov3Ds):
        scales, rotations, visibility_filter = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            scales,
            rotations,
            visibility_filter,
            dL_dcov3Ds,
        )

        dL_dscales, dL_drotations = _C.compute_cov3d_backward(*args)

        grads = (
            dL_dscales,
            dL_drotations,
            None,  # visibility_filter
        )

        return grads

compute_cov3d = _ComputeCov3D.apply

class _ComputeColorFromSH(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        deg,
        dirs,
        shs,
        visibility_filter=None
    ):
        if visibility_filter is None:
            visibility_filter = torch.ones_like(shs[:, 0, 0], dtype=torch.bool)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            deg,
            dirs,
            shs,
            visibility_filter,
        )
        colors, clamped = _C.compute_sh_forward(*args)
        ctx.deg = deg
        ctx.save_for_backward(dirs, shs, visibility_filter, clamped)
        return colors

    @staticmethod
    def backward(ctx, dL_dcolors):
        deg = ctx.deg
        dirs, shs, visibility_filter, clamped = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            deg,
            dirs,
            shs,
            visibility_filter,
            clamped,
            dL_dcolors
        )

        dL_ddirs, dL_dshs = _C.compute_sh_backward(*args)

        grads = (
            None,  # deg
            dL_ddirs,
            dL_dshs,
            None,  # visibility_filter
        )

        return grads

compute_color_from_sh = _ComputeColorFromSH.apply

class _GaussianPreprocess(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug
        )
        depths, radii, means2D, cov3Ds, conics, tiles_touched = _C.preprocess_forward(*args)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(means3D, scales, rotations, cov3Ds_precomp, depths, radii, means2D, cov3Ds, conics, tiles_touched)
        return depths, radii, means2D, cov3Ds, conics, tiles_touched

    @staticmethod
    def backward(ctx, dL_ddepths, dL_dradii, dL_dmeans2D, dL_dcov3Ds, dL_dconics, dL_dtiles_touched):

        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        means3D, scales, rotations, cov3Ds_precomp, depths, radii, means2D, cov3Ds, conics, tiles_touched = ctx.saved_tensors
        
        # Restructure args as C++ method expects them
        args = (means3D, 
                radii, 
                scales, 
                rotations, 
                cov3Ds, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                raster_settings.image_height,
                raster_settings.image_width,
                dL_ddepths,
                dL_dmeans2D,
                dL_dcov3Ds,
                dL_dconics,
                raster_settings.debug)

        dL_dmeans3D, dL_dscales, dL_drotations, dL_dcov3Ds_precomp = _C.preprocess_backward(*args)

        grads = (
            dL_dmeans3D,
            dL_dscales,
            dL_drotations,
            dL_dcov3Ds_precomp,
            None,  # raster_settings
        )
        return grads

gaussian_preprocess = _GaussianPreprocess.apply

class _GaussianRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features,
        depths,  # used for sorting 
        radii, means2D, conics, opacities, tiles_touched,
        visibility_filter,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            features,
            depths, radii, means2D, conics, opacities, tiles_touched,
            visibility_filter,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug
        )
        num_rendered, feature, binningBuffer, imgBuffer = _C.render_forward(*args)
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(features, means2D, conics, opacities, binningBuffer, imgBuffer)
        return feature

    @staticmethod
    def backward(ctx, dL_dfeatures):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        features, means2D, conics, opacities, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (features, means2D, conics, opacities,
                dL_dfeatures, 
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        dL_dfeatures, dL_dmeans2D, dL_dcov3Ds, dL_dconics = _C.render_backward(*args)

        grads = (
            dL_dfeatures,
            None,  # depth
            None,  # grad_radii,
            dL_dmeans2D,
            dL_dcov3Ds,
            dL_dconics,
            None,  # grad_tiles_touched,
            None,  # grad_visibility_filter,
            None,  # raster_settings
        )
        return grads
    
gaussian_render = _GaussianRender.apply

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    debug : bool

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def getProjectionMatrix(fovX, fovY, znear=0.01, zfar=100):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4).cuda()
    
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    return P


def ndc_to_pixel(ndc, size):
    return ((ndc + 1.0) * size - 1.0) * 0.5


if __name__ == "__main__":
    # ======================== Data ========================
    seed = 121
    torch.manual_seed(seed)
    torch.set_printoptions(precision=10)
    
    N = 100

    W = 800
    H = 800

    fovx = 0.6911112070083618
    fovy = 0.6911112070083618

    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)

    viewmat = torch.Tensor([[6.1182e-01,  7.9096e-01, -6.7348e-03,  0.0000e+00], 
                            [ 7.9099e-01, -6.1180e-01,  5.2093e-03,  0.0000e+00], 
                            [ 1.3906e-14, -8.5126e-03, -9.9996e-01,  0.0000e+00], 
                            [ 1.1327e-09,  1.0458e-09,  4.0311e+00,  1.0000e+00]]).cuda()
    projmat = getProjectionMatrix(fovx, fovy).transpose(0, 1)
    full_proj_transform = (viewmat.unsqueeze(0).bmm(projmat.unsqueeze(0))).squeeze(0)

    camparam = torch.Tensor([fx, fy, H/2, W/2]).cuda()
    xyz = torch.randn((N, 3)).cuda() * 2.6 - 1.3

    # generate scale in range of [0, 5]. scale must > 0
    rand_scale = torch.rand(N, 3, device="cuda", dtype=torch.float) * 5 
    rand_quats = torch.rand(N, 4, device="cuda", dtype=torch.float)
    rand_uquats = rand_quats / torch.norm(rand_quats, 2, dim=-1, keepdim=True)
    
    uv, depth = gs.project_point(
        xyz, 
        viewmat, 
        full_proj_transform,
        camparam,
        W, H
    )

    visibility_status = (depth != 0).squeeze(-1)
    
    # ======================== Cov3d ========================
    cov3d = gs.compute_cov3d(rand_scale, rand_uquats)
    cov3d_target = compute_cov3d(rand_scale, rand_uquats)
    torch.testing.assert_close(cov3d, cov3d_target)

    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        viewmatrix=viewmat,
        projmatrix=full_proj_transform,
        debug=False
    )

    depths_target, radii_target, means2D_target, cov3Ds_target, conics_target, tiles_touched_target = gaussian_preprocess(
        xyz,
        rand_scale,
        rand_uquats,
        cov3d,
        raster_settings
    )

    conic, radius, tiles_touched = gs.ewa_project(
        xyz,
        cov3d, 
        viewmat,
        camparam,
        uv, W, H,
        visibility_status)
    
    # print(conic)
    # print(conics_target)
    torch.testing.assert_close(conic, conics_target)
    torch.testing.assert_close(radius, radii_target)
    torch.testing.assert_close(tiles_touched, tiles_touched_target)

# def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = gssplat.GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         debug=pipe.debug
#     )
#     H = viewpoint_camera.image_height
#     W = viewpoint_camera.image_width

#     means3D = pc.get_xyz
#     opacities = pc.get_opacity

#     scales = pc.get_scaling
#     rotations = pc.get_rotation
    
#     visibility_filter = None
#     cov3D_precomp = gssplat.compute_cov3d(scales, rotations, visibility_filter)
#     # scales must be empty if cov3D_precomp is not None
#     scales = torch.Tensor([])
#     rotations = torch.Tensor([])
    
#     depths, radii, means2D, cov3Ds, conics, tiles_touched = gssplat.gaussian_preprocess(
#         means3D,
#         scales,
#         rotations,
#         cov3D_precomp,
#         raster_settings,
#     )
#     means2D.retain_grad()
    
#     # we can choose other mask
#     visibility_filter = radii > 0
        
#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
#     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#     colors = gssplat.compute_color_from_sh(
#         pc.active_sh_degree, 
#         dir_pp_normalized, 
#         pc.get_features,
#         visibility_filter)
    
#     features = colors
    
#     feature = gssplat.gaussian_render(
#         features,
#         depths, radii, means2D, conics, opacities, tiles_touched,
#         visibility_filter,
#         raster_settings,
#     )
    
#     rendered_image = feature[:3]
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": means2D,
#             "visibility_filter" : visibility_filter,
#             "radii": radii}