```bash
pip install .
cd test/gaussian-splatting
mkdir data
# place nerf_synthetic dataset in data/nerf_synthetic
python train.py -s data/nerf_synthetic/lego/ -m output/test --eval

```

See test/gaussian-splatting/gaussian_render/render_split.py for more details

In training, "forward error" represents the error of rendered image between the origin 3dgs diff-gaussian-rasterization implemenation and this implementation in forward pass. "backward error" represents the error of gradient in backward pass.

