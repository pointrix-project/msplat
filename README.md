# DPT-Renderer: Differentiable Point Renderer
The DPT-renderer, full name as differentiable point renderer, serves the backend of POINTRIX. It aims to provide some common underlying operations in differentiable point cloud rendering. Although, only tile based rasterization for 3D Gaussians splatting is supported here, more features will be supported soon.

Visit us at https://github.com/NJU-3DV

Author: NJU-3DV

<!-- ## EWA_Project
```
cd GS_Split
pip install .
cd ..
python test/gs/full_test.py
``` -->


### Usage

```python
### (post-activation) gaussians:
xyzs # [N, 3]
degree # int
shs # [N, C]
opacity # [N, 1]
scales # [N, 3]
rotations # [N, 4], normalized quat

### camera 
# TODO: which convention? colmap as the original paper?
# TODO: what do we need? pose, extrinic, intrinsic in what format?
pose # [4, 4]
fovy # float
H, W # int (render resolution)

### render
import DifferentiablePointRender.GaussianSplatting as gs

# proj points
viewmat = ???
full_proj_transform = ???
camparam = ???
uv, depth = gs.project_point(xyzs, viewmat, full_proj_transform, camparam, W, H)
visibility_status = depth != 0

# cov3d
cov3d = gs.compute_cov3d(scales, rotations, visibility_status) # [N, 6]

# shs to color
viewdirs = ???
color = gs.compute_sh(shs, degree, viewdirs, visibility_status) # [N, 3]

# ewa_project
conic = ewa_project(xyz, cov3d, viewmat, camparam, uv, w, h, visibility_status) # [N, 3]

# sort

# render

### output
image = ??? # or features?
alpha = ???
depth = ???
```