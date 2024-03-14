
import os
import tyro
import tqdm
import imageio
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
import dptr.gs as gs

import kiui
from kiui.mesh import Mesh
from kiui.op import inverse_sigmoid, inverse_softplus
from kiui.cam import orbit_camera, get_perspective

@dataclass
class Options:
    # mesh path
    mesh: str
    # mesh front dir
    front_dir: str = '+z'
    # gaussian numbers
    gs_size: int = 5000
    # gaussian render size
    output_size: int = 512
    # camera radius
    cam_radius: float = 2.4
    # fovy
    fovy: float = 49.1
    # nvdiffrast backend setting
    force_cuda_rast: bool = False


# fit gaussians from mesh
class Model(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.device = torch.device("cuda")

        # mesh renderer
        self.mesh = Mesh.load(self.opt.mesh, bound=0.9, front_dir=self.opt.front_dir)

        if not self.opt.force_cuda_rast:
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        # Gaussian parameters
        N = self.opt.gs_size
        # xyzs = (torch.rand(N, 3, dtype=torch.float32, device=self.device) * 2 - 1) * 0.9
        xyzs = self.mesh.sample_surface(N)
        self.xyzs = nn.Parameter(xyzs)
        rgbs = torch.randn(N, 3, dtype=torch.float32, device=self.device)
        self.rgbs = nn.Parameter(rgbs)
        scales = inverse_softplus(0.01 * torch.ones(N, 3, dtype=torch.float32, device=self.device))
        self.scales = nn.Parameter(scales)
        opacity = inverse_sigmoid(torch.ones(N, 1, dtype=torch.float32, device=self.device) * 0.5)
        self.opacity = nn.Parameter(opacity)
        rot = torch.zeros(N, 4, dtype=torch.float32, device=self.device)
        rot[:, 0] = 1
        self.rot = nn.Parameter(rot)

        # Gaussian activations
        self.rgbs_act = lambda x: torch.sigmoid(x)
        self.scales_act = lambda x: F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x)

        # projection matrix for mesh rendering
        self.proj = torch.from_numpy(get_perspective(self.opt.fovy)).float().to(self.device)

        # intrinsics matrix for gaussian rendering
        focal = 0.5 * float(opt.output_size) / np.tan(0.5 * np.deg2rad(opt.fovy))
        self.intr = torch.tensor([focal, focal, opt.output_size / 2, opt.output_size / 2], dtype=torch.float32, device=self.device)

    @property
    def gaussians(self):
        return (
            self.xyzs,
            self.rgbs_act(self.rgbs),
            self.opacity_act(self.opacity),
            self.scales_act(self.scales), 
            self.rot_act(self.rot),
        )


    @torch.no_grad()
    def render_mesh(self, pose, bg=1):

        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.opt.output_size, self.opt.output_size))

        alpha = (rast[0, ..., 3:] > 0).float()
       
        if self.mesh.vc is not None:
            # use vertex color
            image, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
        else:
            # use texture image
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            image = dr.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]

        # antialias
        image = dr.antialias(image, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        image = alpha * image + (1 - alpha) * bg
        image = image.clamp(0, 1)

        return image
       

    def render_gs(self, pose, bg=1):
        
        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        # OpenGL camera to OpenCV camera
        pose[:3, 1:3] *= -1 # invert up & forward direction

        color = gs.rasterization(
            self.xyzs,
            self.scales_act(self.scales), 
            self.rot_act(self.rot),
            self.opacity_act(self.opacity),
            self.rgbs_act(self.rgbs),
            self.intr,
            torch.inverse(pose),
            self.opt.output_size, self.opt.output_size, 
            bg,
        )

        color = color.permute(1, 2, 0).contiguous()

        return color

    @torch.no_grad()
    def evaluate(self):
        elevation = -15
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        psnrs = []
        for azi in tqdm.tqdm(azimuth):
            pose = orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)
            
            image_gt = self.render_mesh(pose)
            image_pred = self.render_gs(pose)

            psnr = -10 * torch.log10(torch.mean((image_pred - image_gt) ** 2))
            psnrs.append(psnr.item())
            
        psnr = np.mean(psnrs)
        print(f'[Eval] PSNR = {psnr:.4f}')
            

    def fit(self, iters=4096):

        optimizer = torch.optim.Adam([
            {'params': self.xyzs, 'lr': 0.0001},
            {'params': self.rgbs, 'lr': 0.005},
            {'params': self.opacity, 'lr': 0.05},
            {'params': self.scales, 'lr': 0.01},
            {'params': self.rot, 'lr': 0.001},
        ])
        lr_factor = 1

        print(f"[INFO] fitting gaussians...")
        pbar = tqdm.trange(iters)
        for i in pbar:

            ver = np.random.randint(-45, 45)
            hor = np.random.randint(-180, 180)
            rad = self.opt.cam_radius + (np.random.rand() - 0.5)
          
            pose = orbit_camera(ver, hor, rad)
            
            image_gt = self.render_mesh(pose)
            image_pred = self.render_gs(pose)

            loss = F.mse_loss(image_pred, image_gt)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # densify & prune, but keep the total number of Gaussians fixed.
            if i > 0 and i % 256 == 0 and i < iters - 512:
                # prune: select points with opacity thresholding
                opacity = self.opacity_act(self.opacity)
                prune_mask = opacity[:, 0] < 0.01 # [N] bool
                prune_indices = torch.nonzero(prune_mask).squeeze(-1) # [M] long
                num_prune = prune_indices.shape[0]
                if num_prune == 0:
                    continue
                # densify: split points with scale thresholding
                densify_indices = torch.argsort(self.scales.amax(dim=1), descending=True)[:num_prune] # [M] long
                scales = self.scales[densify_indices].repeat(2, 1)
                rot = self.rot[densify_indices].repeat(2, 1)
                xyzs = self.xyzs[densify_indices].repeat(2, 1)
                opacity = self.opacity[densify_indices].repeat(2, 1)
                rgbs = self.rgbs[densify_indices].repeat(2, 1)

                all_indices = torch.cat([prune_indices, densify_indices])
                self.xyzs.data[all_indices] = xyzs
                self.rgbs.data[all_indices] = rgbs
                self.opacity.data[all_indices] = opacity
                self.scales.data[all_indices] = scales
                self.rot.data[all_indices] = rot

                # reset optimizer
                lr_factor *= 0.9
                optimizer = torch.optim.Adam([
                    {'params': self.xyzs, 'lr': 0.0001 * lr_factor},
                    {'params': self.rgbs, 'lr': 0.005},
                    {'params': self.opacity, 'lr': 0.05},
                    {'params': self.scales, 'lr': 0.01},
                    {'params': self.rot, 'lr': 0.001},
                ])

            pbar.set_description(f"MSE = {loss.item():.6f}")
        
        print(f"[INFO] finished fitting gaussians!")
    
    @torch.no_grad()
    def export_video(self, path):
        images = []
        elevation = -15
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):
            pose = orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)
            image = self.render_gs(pose)
            images.append((image.contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.stack(images, axis=0)
        imageio.mimwrite(path, images, fps=30)


if __name__ == '__main__':
    
    opt = tyro.cli(Options)

    model = Model(opt).cuda()
    model.fit()
    model.evaluate()
    model.export_video(os.path.basename(opt.mesh + '.mp4'))