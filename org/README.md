# Differentiable Point Render
The differentiable point renderer is the backend of POINTRIX. Only Tile based Rasterization on 3D Gaussian is supported here. But more features will be supported soon.
Visit us at https://github.com/NJU-3DV
Author: NJU-3DV

```bash
pip install .
cd test/gaussian-splatting

# install origin 3DGS
cd submodules
cd diff-gaussian-rasterization/
pip install .

mkdir data
# place nerf_synthetic dataset in data/nerf_synthetic
python train.py -s /data1/gj/nerf_synthetic/lego -m output/test --eval
```

See test/gaussian-splatting/gaussian_render/render_split.py for more details

In training, "forward error" represents the error of rendered image between the origin 3dgs diff-gaussian-rasterization implemenation and this implementation in forward pass. "backward error" represents the error of gradient in backward pass.

