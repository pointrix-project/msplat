
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import dptr.gs as gs
import imageio
from typing import NamedTuple
from GS_Split import _C as gs_split_c



def compute_sh_torch_impl(sh_coeffs, viewdirs):
    *dims, dim_sh, C = sh_coeffs.shape
    
    bases = eval_sh_bases(dim_sh, viewdirs)
    
    return torch.clamp((bases[..., None] * sh_coeffs).sum(dim=-2) + 0.5, min=0.0)

"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (
                        xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
                    
    return result

def fov2focal(fov, pixels):
    return pixels / (2 * torch.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * torch.arctan(pixels / (2 * focal))

class SimpleCamera:
    def __init__(self, cam_idx, R, T, width=800, height=800, fovY=35.0, bg=0.0):
        
        self.idx = cam_idx
        self.R = R
        self.T = T
        self.width = width
        self.height = height

        # input Fov is in degree, and turned into radian here
        self.fovY = torch.deg2rad(torch.tensor(fovY,dtype=torch.float, device="cuda"))
        self.fovX = focal2fov(fov2focal(self.fovY, height), width)
        
        self.bg = torch.Tensor([bg, bg, bg]).cuda()

        self.world_view_transform = torch.tensor(self.getWorld2View(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = self.getProjectionMatrix(self.fovX, self.fovY).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camparams = torch.Tensor([
            self.width / (2 * np.tan(fovY / 180 * np.pi * 0.5)),
            self.height / (2 * np.tan(fovY / 180 * np.pi * 0.5)),
            self.width / 2, 
            self.height / 2]).cuda().float()
        
        self.tanfovx = np.tan(fovY / 180 * np.pi * 0.5);
        self.tanfovy = np.tan(fovY / 180 * np.pi * 0.5);
        
    @staticmethod
    def getWorld2View(R, t):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return np.float32(Rt)
    
    def get_intrinsics(self):
        focal_x = self.width / (2 * np.tan(self.fovX * 0.5))
        focal_y = self.height / (2 * np.tan(self.fovY * 0.5))

        return torch.tensor([[focal_x, 0, self.width / 2],
                             [0, focal_y, self.height / 2],
                             [0, 0, 1]], device='cuda', dtype=torch.float32)
    
    @staticmethod
    def getProjectionMatrix(fovX, fovY, znear=0.1, zfar=100):
        tanHalfFovY = torch.tan((fovY / 2))
        tanHalfFovX = torch.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P


class GaussianBase(metaclass=ABCMeta):
    """
    TODO: PLY need to be change to .bin file for efficient reading and writing
    Record:
    (1) I have tried using pickle to save it as a bin. It's even bigger.
    (2) TODO: Save it as a json metadata and a bin data.
    """
    def __init__(self):
        self._is_training = False

        self._activations = {
            "scaling": torch.exp,
            "opacity": torch.sigmoid,
            "rotation": torch.nn.functional.normalize,
        }
        self._attributes = {
            "xyz": torch.empty(0),
            "scaling": torch.empty(0),
            "rotation": torch.empty(0),
            "opacity": torch.empty(0),
        }

        self._features = dict()

        # for training
        self._fix_attributions = []
        self._xy = torch.empty(0)

        self.optimizer = None
        self.scheduler_func = dict()

    def load_bin(self, path):
        """_summary_
        :param path: _description_
        :type path: _type_
        :raises ValueError: _description_
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if not any(item in data.keys() for item in self._attributes.keys()):
            raise ValueError(f"Required attributions 'xyz', 'scaling', 'rotation', 'opacity' not found in {path}!")

        for k, v in data.items():
            self._attributes[k] = torch.tensor(v, dtype=torch.float32, device="cuda")
    
    def update_attribute(self, new_attribute):
        self._attributes.update(new_attribute)
    
    @property
    def get_attribute_names(self):
        return self._attributes.keys()
    
    def get_attribute_by_name(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
              return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")
    
    @property
    def get_xy(self):
        return self._xy

    @property
    def get_xyz(self):
        return self.get_attribute_by_name("xyz")
    
    @property
    def get_opacity(self):
        return self.get_attribute_by_name("opacity")
    
    @property
    def get_scaling(self):
        return self.get_attribute_by_name("scaling")

    @property
    def get_rotation(self):
        return self.get_attribute_by_name("rotation")
    
    @property
    def get_features(self):
        # Concat all features in dict into a concat tensor
        cat_features = []
        for value in self._features.values():
            cat_features.append(value)

        return torch.cat(cat_features, dim=-1)
    
    @ property
    def get_features_dim(self):
        features_dim = dict()
        for key, val in self._features.items():
            features_dim[key] = val.shape[-1]
        
        return features_dim
    
    @abstractmethod
    def set_feature(self):
        pass

    def train(self):
        for attribute_name in self._attributes.keys():
            if attribute_name in self._fix_attributions:
                self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(False)
            else:
                self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self._xy = torch.zeros_like(self._attributes["xyz"], requires_grad=True, device="cuda")
        self._xy.retain_grad()

        self._is_training = True
    
    def set_optimizer(self, params:dict):
        l = []
        for k, v in self._attributes.items():
            pass

    def step(self):
        if self.is_training:
            if self.optimizer is None:
                raise ValueError("The optimizer should be set.")
            self.optimizer.step()
            self.optimizer.zero_grad()


from attrikit import _C

class sh2rgb(torch.autograd.Function):
    """Compute spherical harmonics
    Args:
        points_shs (Tensor): harmonic coefficients of points.
        viewdirs (Tensor): viewing directions.
        degree (Int): degree of SHs.
    """
    @staticmethod
    def forward(ctx, points_shs, view_dirs, degree):
        ctx.degree = degree
        ctx.save_for_backward(points_shs, view_dirs)
        return _C.sh2rgb(points_shs, view_dirs, degree)

    @staticmethod
    def backward(ctx, grad_color):
        points_shs, view_dirs = ctx.saved_tensors
        degree = ctx.degree
        grad_points_shs, grad_view_dirs = _C.sh2rgb_backward(
            points_shs, grad_color, view_dirs, degree)
        grads = (grad_points_shs, grad_view_dirs, None)
        return grads


class GaussianSplat(GaussianBase):
    def __init__(self):
        super(GaussianSplat, self).__init__()
        extra_attributes = {
            "shs_dc": torch.empty(0),
            "shs_rest": torch.empty(0),
        }

        self.update_attribute(extra_attributes)
    
    def set_feature(self, camera):
        shs_dc = self.get_attribute_by_name("shs_dc")
        
        # self._features["rgb"] = shs_dc.squeeze(1) * 0.28209479177387814 + 0.5
        # self._features["rgb"] = torch.clamp(self._features["rgb"], min=0, max=1)
        
        # direction = (self.get_xyz - camera.camera_center.repeat(self.get_xyz.shape[0], 1))
        # self._features["rgb"] = gs.compute_sh(shs_dc, 0, direction)
        
        shs_rest = self.get_attribute_by_name("shs_rest")
        direction = (self.get_xyz - camera.camera_center.repeat(self.get_xyz.shape[0], 1))
        direction = direction / direction.norm(dim=1, keepdim=True)
        shs = torch.cat((shs_dc, shs_rest), dim=1)
        
        print("gs")
        self._features["rgb"] = gs.compute_sh(shs, 3, direction)
        
        # print("attrikit")
        # self._features["rgb"] = sh2rgb.apply(shs, direction, 3)
        
        print(self._features["rgb"].shape)

# GS_Split
class _GaussianRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features,
        depths,  # used for sorting 
        radii, 
        means2D, 
        conics, 
        opacities, 
        tiles_touched,
        visibility_filter,
        raster_settings,
    ):
        if visibility_filter is None:
            visibility_filter = torch.ones_like(features[:, 0], dtype=torch.bool)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            features,
            depths, radii, means2D, conics, opacities, tiles_touched,
            visibility_filter,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug
        )
        num_rendered, feature, binningBuffer, imgBuffer = gs_split_c.render_forward(*args)
        
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

        dL_dfeatures, dL_dmeans2D, dL_dcov3Ds, dL_dconics = gs_split_c.render_backward(*args)

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
        cov3Ds = gs_split_c.compute_cov3d_forward(*args)
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

        dL_dscales, dL_drotations = gs_split_c.compute_cov3d_backward(*args)

        grads = (
            dL_dscales,
            dL_drotations,
            None,  # visibility_filter
        )

        return grads

compute_cov3d = _ComputeCov3D.apply


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
        depths, radii, means2D, cov3Ds, conics, tiles_touched = gs_split_c.preprocess_forward(*args)
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

        dL_dmeans3D, dL_dscales, dL_drotations, dL_dcov3Ds_precomp = gs_split_c.preprocess_backward(*args)

        grads = (
            dL_dmeans3D,
            dL_dscales,
            dL_drotations,
            dL_dcov3Ds_precomp,
            None,  # raster_settings
        )
        return grads

gaussian_preprocess = _GaussianPreprocess.apply

if __name__ == "__main__":
    bin_file = "./data/lego.bin"
    
    print(os.path.abspath(bin_file))
    
    world = GaussianSplat()
    world.load_bin(bin_file)

    R = np.array([[-0.9999999403953552, -0.0, -0.0],
                  [0.0, 0.7341099977493286, -0.6790305972099304],
                  [-0.0, -0.6790306568145752, -0.7341098785400391]], dtype=np.float32)
    T = np.array([-0.0, 2.7372601032257085, 2.9592916965484624],
                 dtype=np.float32)
    T = -np.matmul(R, T)
    
    bg = 0.0

    camera = SimpleCamera(0, R, T)
    world.set_feature(camera)
    
    viewmat = camera.world_view_transform
    projmat = camera.full_proj_transform
    camparam = camera.camparams
    
    H = camera.height
    W = camera.width
    
    setting = GaussianRasterizationSettings
    setting.image_height = camera.height
    setting.image_width = camera.width
    setting.tanfovx = camera.tanfovx
    setting.tanfovy = camera.tanfovy
    setting.viewmatrix = viewmat
    setting.projmatrix = projmat
    setting.debug = False
    
    print("view:\n", viewmat)
    print("proj:\n", projmat)
    print("cam:\n", camparam)
    print("xyz:\n", torch.min(world.get_xyz, dim=0), " ", torch.max(world.get_xyz, dim=0))
    print("scale:\n", torch.min(world.get_scaling, dim=0), " ", torch.max(world.get_scaling, dim=0))
    
    with torch.no_grad():
        (
            uv,
            depth 
        ) = gs.project_point(
            world.get_xyz, 
            viewmat, 
            projmat,
            camparam,
            W, H)
        
        visibility_status = depth != 0
        
        # compute cov3d
        cov3d = gs.compute_cov3d(
            world.get_scaling, 
            world.get_rotation, 
            visibility_status)
        
        # print("GS_cov3d")
        # cov3d = compute_cov3d(
        #     world.get_scaling, 
        #     world.get_rotation
        # )
        
        # print("Preprocess GS")
        # (
        #     depth, 
        #     radius, 
        #     uv, 
        #     cov3d,
        #     conic,
        #     tiles_touched
        # ) = gaussian_preprocess(
        #     world.get_xyz, 
        #     world.get_scaling, 
        #     world.get_rotation, 
        #     cov3d, 
        #     setting
        # )
        
        # ewa project
        (
            conic, 
            radius, 
            tiles_touched
        ) = gs.ewa_project(
            world.get_xyz,
            cov3d, 
            viewmat,
            camparam,
            uv,
            W, H,
            visibility_status
        )
        
        # print("Render GS")
        # # feature = torch.cat([depth, depth, depth], dim=-1)
        # render_feature = gaussian_render(
        #     world.get_features,
        #     depth,
        #     radius,
        #     uv,
        #     conic,
        #     world.get_opacity,
        #     tiles_touched,
        #     None,
        #     setting
        # )
        
        # sort
        (
            gaussian_idx_sorted, 
            tile_bins
        ) = gs.sort_gaussian(
            uv, 
            depth, 
            W, H, 
            radius, 
            tiles_touched
        )
        
        # alpha blending
        # feature = torch.cat([depth, depth, depth], dim=-1)
        feature = world.get_features
        # feature = torch.cat([world.get_features[:, 0:1], world.get_features[:, 0:1], world.get_features[:, 0:1]], dim=-1)
        (
            render_feature, 
            final_T, 
            ncontrib
        ) = gs.alpha_blending(
            uv, 
            conic, 
            world.get_opacity, 
            feature,
            gaussian_idx_sorted, 
            tile_bins, 
            bg, 
            W, 
            H
        )
        
        print(render_feature.shape)
        
        import matplotlib.pyplot as plt
        plt.imshow(render_feature.detach().permute(1, 2, 0).cpu().numpy())
        plt.savefig("lego.png")
        # save_img = (render_feature.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # imageio.imwrite("lego.png", save_img)
        