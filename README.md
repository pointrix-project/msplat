# DPTR: Differentiable PoinT Renderer
<!-- ```
Differentiable PoinT Renderer, backend for POINTRIX.
 ____  ____ _____ ____  
|  _ \|  _ \_   _|  _ \ 
| | | | |_) || | | |_) |
| |_| |  __/ | | |  _ < 
|____/|_|    |_| |_| \_\
``` -->
<p align="center">
    <a href="">🌐 <b> Project Page </b> </a> | 
    <a href="">📰 <b> Document </b> </a>
</p>


The **D**ifferentiable **P**oin**T** **R**enderer (**DPTR**), serves as the backend of [POINTRIX]() and is designed to offer foundational functionalities for differentiable point cloud rendering. Presently, DPTR exclusively supports tile-based 3D Gaussian Splatting rasterization. However, the roadmap includes the incorporation of additional point-based rendering primitives.
![dptr](media/media.gif)
The logo of [DPTR](https://www.bing.com/images/create/a-logo-for-dptr-differentiable-point-renderer2c-w/1-65d4bedd4ab84dc2a0983d1137a2ae75?id=Aq9grH0ZLohkId7qqRf3xQ%3d%3d&view=detailv2&idpp=genimg&thId=OIG1.JGMYYbQ9W7pdur2USXGO&FORM=GCRIDP&mode=overlay) is desisned by [Microsoft Designer](https://designer.microsoft.com/).

## How to install
1. Install with source
```shell
# clone the code from github
git clone https://github.com/NJU-3DV/DPTR.git
cd DPTR
# install dptr
pip install .
```

2. Install with pip
```shell
pip install dptr
```

## Tutorial: Fitting the logo with 3D Gaussian Splatting
In this tutorial, we will demonstrate how to use DPTR to implement 3D Gaussian Splatting (3DGS) to fit the DPTR logo step by step. If you are not familiar with 3DGS, you can learn more about it through the original [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) project.

### Create a simple colorful 3D Gaussian class
First, we create a simplified colorful 3D Gaussian point cloud. The attributes we set include 3D position, scale, rotation, opacity, and RGB color, all of which are randomly initialized.
```python
class SimpleGaussian:
    def __init__(self, num_points=100000):
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   torch.rand((N, 4), dtype=torch.float32).cuda(),
            "opacity":  torch.rand((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }
```

Next, we set activation functions for each attribute to ensure that they always remain within reasonable bounds. For the scale, which must be greater than 0, we use the absolute values, not exponential functions, which are too steep. For the RGB color and opacity, which are within the range of [0, 1], we choose the sigmoid function. As for rotation, represented using unit quaternions where the magnitude must be 1, we use normalization function.

```python
        self._activations = {
            "scale": lambda x: torch.abs(x) + 1e-8,
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.sigmoid,
            "rgb": torch.sigmoid
        }
```

To perform gradient-based optimization, we allow gradient computation for each attribute by setting *requires_grad_* to *true* and create an optimizer.
```python
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=0.01)
```

We encapsulate each backward step and gradient zeroing into a function.
```python
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
```

Then, we need a function to retrieve the attributes of the 3D Gaussian, returning the corresponding activated attributes according to the name.
```python
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")
```

### Read the target logo image
Read the logo image, normalize it, and then convert it into a tensor with a shape of [C, H, W].
```python
    image_file = "./media/dptr.png"
    img = np.array(Image.open(image_file))
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1)
    
    C, H, W = gt.shape
```

### Set a Camera
In DPTR, the camera intrinsic parameters are represented by a tensor of shape [4], consisting of [fx, fy, cx, cy]. Here, fx and fy denote the focal lengths of the camera (in pixels), while cx and cy represent the offset of the principal point of the camera, relative to the center of the top-left pixel of the image.
```python
    bg = 0
    fov_x = math.pi / 2.0
    fx = 0.5 * float(W) / math.tan(0.5 * fov_x)
    camparam = torch.Tensor([fx, fx, float(W) / 2, float(H) / 2]).cuda().float()
    viewmat = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 8.0, 1.0]]).cuda().float()
    projmat = viewmat.clone()
```

### Train
Create a 3D Gaussian point cloud and optimize it!
```python
    gaussians = SimpleGaussian(num_points=100000)

    max_iter = 2000
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    mse_loss = nn.MSELoss()

    for iteration in range(0, max_iter):
        
        rendered_feature = gs.rasterization(
            gaussians.get_attribute("xyz"),
            gaussians.get_attribute("scale"),
            gaussians.get_attribute("rotate"), 
            gaussians.get_attribute("opacity"),
            gaussians.get_attribute("rgb"),
            viewmat,
            projmat,
            camparam,
            W, H, bg)
        
        loss = mse_loss(render_feature, gt)
        loss.backward()
        
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 20 == 0:
            show_data = render_feature.detach().permute(1, 2, 0)
            show_data = torch.clamp(show_data, 0.0, 1.0)
            frames.append((show_data.cpu().numpy() * 255).astype(np.uint8))
    
    progress_bar.close()
```

We then save the results in the optimization process as a GIF image.
```python
    frames = [Image.fromarray(frame) for frame in frames]
    out_dir = "media"
    os.makedirs(out_dir, exist_ok=True)
    frames[0].save(
        f"{out_dir}/tutorial.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,
    )
```

The complete code for this tutorial can be found in: "./tutorials/gs.py". And you could easily run it by:
```shell
python tutorials/gs.py
```

## Plans
- [ ] Further enhance interface user-friendliness.
- [ ] Optimization on camera.
- [ ] Higher order spherical harmonic.
- [ ] Not only 3-channal SHs.
- [ ] Spherical Gaussian.
