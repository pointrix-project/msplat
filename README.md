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

![dptr](media/dptr_w.png#gh-light-mode-only)
![dptr](media/dptr_b.png#gh-dark-mode-only)

The logo of [DPTR](https://www.bing.com/images/create/a-minimalist-logo-with-a-solid-white-background-th/1-65dc22883b234064b70d857744a00e96?id=Jl8gopEgQ7udGtZyYZjIIg%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG3.iiX1JtCk02kNJ_Zn5ORG&FORM=SYDBIC) is desisned by [Microsoft Designer](https://designer.microsoft.com/). The front is Potra Font, designed by Alejo Bergmann.


## How to install
1. Install from source
```shell
# clone the code from github
git clone https://github.com/NJU-3DV/DPTR.git --recursive
cd DPTR
# install dptr
pip install .
```

2. Install from pip (Not yet.)
```shell
pip install dptr
```

## Camera Model
DPTR use a pinhole camera model:

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

where, $(X, Y, Z)$ are the coordinate of a 3D point in the world coordinate system, $(u,v)$ are the coordinate of the projection point in pixels, $d$ is the depth value. $K$ is a matrix of intrinsic parameters, $R_{cw}$ and $T_{cw}$ are extrinsic parameters.  $(c_x, c_y)$ is the principal point defined in the image coordinate system with the upper left corner of the image as the origin. In CG community, the principal point is usually set at the image center, i.e. $(c_x, c_y)=(\frac{float(W)}{2.0},\frac{float(H)}{2.0})$. $f_x$ and $f_y$ are the focal lengths in pixels.

- Intrinsic Parameters $[f_x, f_y, c_x, c_y]$
- Extrinsic Parameters $[R_{cw}|t_{cw}]$


## Tutorial: Fitting the logo with 3D Gaussian Splatting
In this tutorial, we will demonstrate how to use DPTR to implement 3D Gaussian Splatting (3DGS) to fit the DPTR logo step by step. If you are not familiar with 3DGS, you can learn more about it through the original [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) project.

### Create a simple colorful 3D Gaussian class
First, we create a simplified class for colorful 3D Gaussian point cloud. The attributes we set include 3D position, scale, rotation, opacity, and RGB color, all of which are randomly initialized.
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
Set a camera.
```python
    bg = 1
    fov = math.pi / 2.0
    fx = 0.5 * float(W) / math.tan(0.5 * fov)
    fy = 0.5 * float(H) / math.tan(0.5 * fov)
    intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float()
    extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 4.0]]).cuda().float()
```

### Train
Create a 3D Gaussian point cloud and optimize it!
```python
    gaussians = SimpleGaussian(num_points=100000)

    max_iter = 10000
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    cal_loss = nn.SmoothL1Loss()

    for iteration in range(0, max_iter):
        
        rendered_feature = gs.rasterization(
            gaussians.get_attribute("xyz"),
            gaussians.get_attribute("scale"),
            gaussians.get_attribute("rotate"), 
            gaussians.get_attribute("opacity"),
            gaussians.get_attribute("rgb"),
            intr,
            extr,
            W, H, bg)
        
        loss = cal_loss(rendered_feature, gt)
        loss.backward()
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 50 == 0:
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
- [x] Optimization on camera.
- [x] Change camera to cv style.
- [ ] Higher order spherical harmonic.
- [ ] Not only 3-channal SHs.
- [ ] Spherical Gaussian.
- [ ] Further enhance interface user-friendliness.