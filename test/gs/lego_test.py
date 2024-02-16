
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import dptr.gs as gs
import imageio


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
        shs_rest = self.get_attribute_by_name("shs_rest")
        direction = (self.get_xyz - camera.camera_center.repeat(self.get_xyz.shape[0], 1))
        direction = direction / direction.norm(dim=1, keepdim=True)
        shs = torch.cat((shs_dc, shs_rest), dim=1)
        
        self._features["rgb"] = gs.compute_sh(shs, 3, direction)

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
        
        # ewa project
        (
            conic, 
            radius, 
            tiles
        ) = gs.ewa_project(
            world.get_xyz,
            cov3d, 
            viewmat,
            camparam,
            uv,
            W, H,
            visibility_status
        )
        
        # sort
        (
            idx_sorted, 
            tile_bins
        ) = gs.sort_gaussian(
            uv, 
            depth, 
            W, H, 
            radius, 
            tiles
        )
        
        # alpha blending
        # feature = torch.cat([depth, depth, depth], dim=-1)
        feature = world.get_features
        (
            render_feature, 
            final_T, 
            ncontrib
        ) = gs.alpha_blending(
            uv, 
            conic, 
            world.get_opacity, 
            feature,
            idx_sorted, 
            tile_bins, 
            bg, 
            W, 
            H
        )
        
        show_image = render_feature.detach()
        show_image = torch.clamp(show_image, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        show_image = (show_image * 255).astype(np.uint8)
        imageio.imwrite("lego.png", show_image)
        