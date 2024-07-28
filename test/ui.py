
import time
import torch
import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from msplat import rasterization, compute_sh
from plyfile import PlyData, PlyElement

try:
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    import_3dgs = True
except:
    import_3dgs = False


class GaussianPoints:
    def __init__(self):
        self.activate_func = {
            "opacity": torch.sigmoid,
            "scale": torch.exp,
            "rotate": torch.nn.functional.normalize
        }

        self.attribute = {
            "xyz": torch.empty(0),
            "opacity": torch.empty(0),
            "scale": torch.empty(0),
            "rotate": torch.empty(0),
            "shs": torch.empty(0) 
        }
    
    def __getitem__(self, name):
        assert name in self.attribute.keys(), f"No attribute named as {name}."

        if name in self.activate_func.keys():
            return self.activate_func[name](self.attribute[name])
        else:
            return self.attribute[name]
    
    @property
    def size(self):
        return self.attribute["xyz"].shape[0]

    def load_from_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., None]

        scale = np.zeros((xyz.shape[0], 3))
        scale[:, 0] = np.asarray(plydata.elements[0]["scale_0"])
        scale[:, 1] = np.asarray(plydata.elements[0]["scale_1"])
        scale[:, 2] = np.asarray(plydata.elements[0]["scale_2"])

        rotate = np.zeros((xyz.shape[0], 4))
        rotate[:, 0] = np.asarray(plydata.elements[0]["rot_0"])
        rotate[:, 1] = np.asarray(plydata.elements[0]["rot_1"])
        rotate[:, 2] = np.asarray(plydata.elements[0]["rot_2"])
        rotate[:, 3] = np.asarray(plydata.elements[0]["rot_3"])

        shs = np.zeros((xyz.shape[0], 3, 16))
        shs[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        for i in range(15):
            shs[:, 0, i + 1] = np.asarray(plydata.elements[0][f"f_rest_{i + 0}"])
            shs[:, 1, i + 1] = np.asarray(plydata.elements[0][f"f_rest_{i + 15}"])
            shs[:, 2, i + 1] = np.asarray(plydata.elements[0][f"f_rest_{i + 30}"])
        
        self.attribute["xyz"] = torch.from_numpy(xyz).float().cuda()
        self.attribute["opacity"] = torch.from_numpy(opacity).float().cuda()
        self.attribute["scale"] = torch.from_numpy(scale).float().cuda()
        self.attribute["rotate"] = torch.from_numpy(rotate).float().cuda()
        self.attribute["shs"] = torch.from_numpy(shs).float().cuda()


class GUI:
    def __init__(self, ply_file, frame_size, intr, extr, cam_type="PERSPECTIVE", mode="rgb", backend="3dgs"):

        print("Backebd: ", backend)

        self.gaussians = GaussianPoints()
        self.gaussians.load_from_ply(ply_file)

        self.W, self.H = frame_size
        self.intr = torch.tensor(intr).cuda().float()
        self.extr = torch.tensor(extr).cuda().float()

        assert cam_type in ["PERSPECTIVE", "ORTHOGRAPHIC"]
        self.cam_type = cam_type

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        assert mode in ["rgb", "normal", "depth"]
        self.mode = mode

        assert backend in ["3dgs", "msplat"]
        self.backbend = backend

        self.buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)

        dpg.create_context()
        self.register_dpg()

        self.fps_list = [] # for stable fps

        if import_3dgs and backend == "3dgs":
            # prepare for 3dgs
            def intr2fov(intr, width, height):
                fovx = 2 * math.atan(width/(2*intr[0]))
                fovy = 2 * math.atan(height/(2*intr[1]))

                return fovx, fovy

            def getProjMatrix(fovX, fovY, znear=0.01, zfar=100.0):
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
            
            fovx, fovy = intr2fov(intr, self.W, self.H)
            self.tanfovx = math.tan(fovx * 0.5)
            self.tanfovy = math.tan(fovy * 0.5)

            self.world_view_transform = self.extr.transpose(0, 1)
            projection_matrix = getProjMatrix(fovx, fovy).transpose(0,1)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

            self.bg = torch.tensor([0, 0, 0], device="cuda", dtype=torch.float)


    def __del__(self):
        dpg.destroy_context()
    
    def vis_buffer(self):
        if self.mode == "depth":
            valid_mask = self.buffer > 0
            valid_values = self.buffer[valid_mask]

            if len(valid_values) == 0:
                raise ValueError("No valid depth values found in the input buffer.")

            min_valid = valid_values.min()
            max_valid = valid_values.max()
            normalized_depth = np.zeros_like(self.buffer, dtype=np.float32)
            normalized_depth[valid_mask] = (valid_values - min_valid) / (max_valid - min_valid)

            cmap = plt.get_cmap('jet')
            colored_depth = np.zeros((*normalized_depth.shape, 3), dtype=np.float32)
            colored_depth[valid_mask] = cmap(normalized_depth[valid_mask])[:, :3]

            self.buffer = colored_depth

        elif self.mode  == "normal":
            self.buffer  = self.buffer  * 0.5 + 0.5

        self.buffer = np.ascontiguousarray(self.buffer)
    
    def render(self):
        self.step()
        dpg.render_dearpygui_frame()
    
    @torch.no_grad()
    def render_with_3dgs(self):
        if not import_3dgs:
            raise ImportError("Module diff_gaussian_rasterization is not found.")
        
        center = - (self.extr[:3, :3]) @ self.extr[:3, -1:]

        raster_settings = GaussianRasterizationSettings(
            image_height=self.H,
            image_width=self.W,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=self.world_view_transform,
            projmatrix=self.full_proj_transform,
            sh_degree=3,
            campos=center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        render, _ = rasterizer(
            means3D = self.gaussians["xyz"],
            means2D = None,
            shs = self.gaussians["shs"].permute(0, 2, 1),
            colors_precomp = None,
            opacities = self.gaussians["opacity"],
            scales = self.gaussians["scale"],
            rotations = self.gaussians["rotate"],
            cov3D_precomp = None)
        
        return render

    @torch.no_grad()
    def render_with_msplat(self):
        center = - (self.extr[:3, :3]) @ self.extr[:3, -1:]

        direction = (self.gaussians["xyz"] - center.squeeze(-1)[None])
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = compute_sh(self.gaussians["shs"], direction) + 0.5

        render = rasterization(
            self.gaussians["xyz"], 
            self.gaussians["scale"], 
            self.gaussians["rotate"], 
            self.gaussians["opacity"], 
            rgb, 
            self.intr, 
            self.extr, 
            self.W,
            self.H, 
            0.0
        )

        return render
    
    def step(self):
        self.start.record()

        # render
        if self.backbend == "msplat":
            render = self.render_with_msplat()
        elif self.backbend == "3dgs":
            render = self.render_with_3dgs()
        
        self.end.record()
        torch.cuda.synchronize()

        t = self.start.elapsed_time(self.end)
        fps = int(1000 / t+1e-8)

        if len(self.fps_list) > 10:
            self.fps_list.pop(0)    
        self.fps_list.append(fps)

        fps = int(np.mean(self.fps_list))

        self.buffer = render.permute(1, 2, 0).cpu().numpy()
        self.vis_buffer()

        dpg.set_value("_log_infer_time", f'{t:.4f}ms ({fps} FPS)')
        dpg.set_value("_texture", self.buffer)
    
    def register_dpg(self):

        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        ## control windows
        with dpg.window(label="Control", tag="_control_window", width=300, height=20):
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

        dpg.create_viewport(title='Viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()


if __name__ == "__main__":

    ply_file = "../data/lego.ply"

    frame_size = [800, 800] 
    intr = [1000, 1000, 400, 400]

    extr = [[6.9778e-01, -7.1631e-01,  2.9802e-08,  1.5128e-07], 
            [-2.0250e-01, -1.9726e-01, -9.5921e-01,  4.2758e-08], 
            [6.8709e-01,  6.6932e-01, -2.8269e-01,  4.0311e+00],
            [0, 0, 0, 1]]
    windows = GUI(ply_file, frame_size, intr, extr, mode="rgb", backend="3dgs")

    while True:
        windows.render()
