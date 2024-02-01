from typing import NamedTuple
import torch

import os
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
