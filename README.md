

# MSplat

MSplat is a modular differential gaussian rasterization library. We have refactored the original repository for easier understanding and extension.

**What's new：**
- The camera model has been changed from FOV to pinhole.
- Optimization on camera model is support.
- SH evaluation is now supported up to level 10.
- Differentiable rendering of extensible features (RGB, depth and even more) is now supported. 

## How to install
```shell
git clone https://github.com/pointrix-project/msplat.git --recursive
cd msplat
pip install .
```

## Camera Model
We use a pinhole camera model:

$$dx=K[R_{cw}|t_{cw}]X$$

$$d\begin{bmatrix}
  u \\ 
  v \\ 
  1
\end{bmatrix} = \begin{bmatrix}
  f_x & 0 & c_x \\
  0 &  f_y & c_y \\
  0 & 0 & 1 \\
\end{bmatrix}\begin{bmatrix}
  r_{11} & r_{12} & r_{13} & t_1 \\
  r_{21} & r_{22} & r_{23} & t_2 \\
  r_{31} & r_{32} & r_{33} & t_3 \\
\end{bmatrix}\begin{bmatrix}
  X \\ 
  Y \\ 
  Z \\ 
  1
\end{bmatrix}$$

* $(X, Y, Z)$ is the coordinate of a 3D point in the world coordinate system.
* $(u,v)$ are the coordinate of the projection point in **pixels**, $d$ is the depth value. 
* $K$ is a matrix of intrinsic parameters in pixels. $(c_x, c_y)$ is the principal point, which is usually the image center $(\frac{W}{2},\frac{H}{2})$. $f_x$ and $f_y$ are the focal lengths.
* $R_{cw}$ and $T_{cw}$ are extrinsic parameters (view matrix) indicated **world-to-camera** transformation in  [**OpenCV/Colmap** convention](https://kit.kiui.moe/camera/#common-camera-coordinate-systems).

In our API, you need to provide the camera parameters in the following format:
- Intrinsic Parameters $[f_x, f_y, c_x, c_y]$
- Extrinsic Parameters $[R_{cw}|t_{cw}]$

## Optimization on Camera
It hasn't been validated too much. Since a camera is involved in the optimization of many 3D Gaussian points, the gradient of the camera is  accumulating, potentially leading to numerical issues.

## High-level SH
We have upgraded the level of the Spherical Harmonics (SH) from the original 3 to 10.

## Extensiable features
We abstract concepts such as RGB, depth, etc. to features and enable feature rendering for any channels. As a result, we easily render rgb map, depth map and orther feature maps without the need to modify and recompile the kernel.

## More easy to understand
We break the heavy and confusing kernel to clear and easy-understant parts, which is more friendly for beginers. You can invoke the renderer with a single line of command, or you can break it down into steps.

```python
import torch
import gs

# the inputs
H: int = ... # image height
W: int = ... # image width
intr: torch.Tensor = ... # [4,], (fx, fy, cx, cy) in pixels
extr: torch.Tensor = ... # [4, 4], camera extrinsics in OpenCV convention.
xyzs: torch.Tensor = ... # [N, 3], Gaussian centers
feats: torch.Tensor = ... # [N, C], Gaussian RGB features (or any other features with arbitrary channels!)
opacity: torch.Tensor = ... # [N, 1], Gaussian opacity
scales: torch.Tensor = ... # [N, 3], Gaussian scales
rotations: torch.Tensor = ... # [N, 4], Gaussian rotations in quaternion
bg: float = 0 # scalar, background for rendered feature maps

# =================== one-line interface =================== 
rendered_image = gs.rasterization(
    xyzs, scales, rotations, opacity, feats, intr, extr, H, W, bg
) # [C, H, W]

# =================== steps interface =================== 
# project points
(uv, depth) = gs.project_point(xyzs, intr, extr, W, H)
visible = depth != 0

# compute cov3d
cov3d = gs.compute_cov3d(scales, rotations, visible)

# ewa project
(conic, radius, tiles_touched) = gs.ewa_project(
    xyzs, cov3d, intr, extr, uv, W, H, visible
)

# sort
(gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
    uv, depth, W, H, radius, tiles_touched
)

# alpha blending
rendered_image = gs.alpha_blending(
    uv, conic, opacity, rgbs, gaussian_ids_sorted, tile_range, bg, W, H,
)
```

## Acknowledgment
We express our sincerest appreciation to the developers and contributors of the following projects:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): 3D Gaussian Splatting for Real-Time Radiance Field Rendering.

If you find that 3D gaussian splatting implemetation in our project helpful, please consider to cite:
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```