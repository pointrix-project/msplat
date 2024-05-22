
import time
import math
import torch
import msplat as ms

BLOCK_X = 16
BLOCK_Y = 16


def ewa_project_torch_impl(
    xyz,
    cov3d, 
    intr,
    extr,
    xy,
    W, H,
    visible
):
    Wmat = extr[..., :3, :3]
    p = extr[..., :3, 3]

    t = torch.matmul(Wmat, xyz[..., None])[..., 0] + p
    rz = 1.0 / t[..., 2]
    rz2 = rz**2
    
    fx = intr[..., 0]
    fy = intr[..., 1]
    
    Jmat = torch.stack(
        [
            torch.stack([fx * rz, torch.zeros_like(rz), -fx * t[..., 0] * rz2], dim=-1),
            torch.stack([torch.zeros_like(rz), fy * rz, -fy * t[..., 1] * rz2], dim=-1),
        ],
        dim=-2,
    )
    
    T = Jmat @ Wmat
    cov3d_1 = torch.stack([cov3d[..., 0], cov3d[..., 1], cov3d[..., 2]], dim=-1)
    cov3d_2 = torch.stack([cov3d[..., 1], cov3d[..., 3], cov3d[..., 4]], dim=-1)
    cov3d_3 = torch.stack([cov3d[..., 2], cov3d[..., 4], cov3d[..., 5]], dim=-1)
    cov3d = torch.stack([cov3d_1, cov3d_2, cov3d_3], dim=-1)
    
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    cov2d[..., 0, 0] = cov2d[:, 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[:, 1, 1] + 0.3

    # Compute extent in screen space
    det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2
    det_mask = det != 0
        
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )
    
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))
    
    # get tiles
    top_left = torch.zeros_like(xy, dtype=torch.int, device=uv.device)
    bottom_right = torch.zeros_like(xy, dtype=torch.int, device=uv.device)
    top_left[:, 0] = ((xy[:, 0] - radius) / BLOCK_X)
    top_left[:, 1] = ((xy[:, 1] - radius) / BLOCK_Y)
    bottom_right[:, 0] = ((xy[:, 0] + radius + BLOCK_X - 1) / BLOCK_X)
    bottom_right[:, 1] = ((xy[:, 1] + radius + BLOCK_Y - 1) / BLOCK_Y)
    
    tile_bounds = torch.zeros(2, dtype=torch.int, device=uv.device)
    tile_bounds[0] = (W + BLOCK_X - 1) / BLOCK_X
    tile_bounds[1] = (H + BLOCK_Y - 1) / BLOCK_Y
    
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    
    tiles_tmp = tile_max - tile_min
    tiles = tiles_tmp[..., 0] * tiles_tmp[..., 1]
    
    mask = torch.logical_and(tiles != 0, det_mask)
    mask = torch.logical_and(visible, mask)
    
    conic = torch.nan_to_num(conic)
    radius = torch.nan_to_num(radius)
    tiles = torch.nan_to_num(tiles)

    conic = conic * mask.float()[..., None]
    radius = radius * mask.float()
    tiles = tiles * mask.float()
    
    return conic, radius.int(), tiles.int()


def getProjectionMatrix(fovX, fovY, znear=0.01, zfar=100):
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

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def ndc_to_pixel(ndc, size):
    return ((ndc + 1.0) * size - 1.0) * 0.5

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    
    iters = 1
    N = 10000
    
    print("=============================== running test on ewa_project ===============================")
    
    W, H = 800, 800
    fx, fy = 1111, 1111
    
    intr = torch.Tensor([fx, fy, float(H) / 2, float(W/2) / 2]).cuda()
    extr = torch.Tensor([[ 6.1182e-01,  7.9099e-01,  1.3906e-14,  1.1327e-09],
                         [ 7.9096e-01, -6.1180e-01, -8.5126e-03,  1.0458e-09],
                         [-6.7348e-03,  5.2093e-03, -9.9996e-01,  4.0311e+00]]).cuda()
    
    xyz = torch.randn((N, 3), dtype=torch.float32).cuda() * 2.6 - 1.3
    
    # generate scale in range of [1, 2]. scale must > 0
    rand_scale = torch.rand(N, 3, device="cuda", dtype=torch.float) + 1
    rand_quats = torch.rand(N, 4, device="cuda", dtype=torch.float)
    rand_uquats = rand_quats / torch.norm(rand_quats, 2, dim=-1, keepdim=True)
    
    cov3d = ms.compute_cov3d(rand_scale, rand_uquats)
    
    # project points
    (
        uv,
        depth 
    ) = ms.project_point(
        xyz, 
        intr, 
        extr,
        W, H)

    visible = (depth != 0).squeeze(-1)

    xyz1 = xyz.clone().requires_grad_()
    xyz2 = xyz.clone().requires_grad_()
    cov3d1 = cov3d.clone().requires_grad_()
    cov3d2 = cov3d.clone().requires_grad_()
    intr1 = intr.clone().requires_grad_()
    intr2 = intr.clone().requires_grad_()
    extr1 = extr.clone().requires_grad_()
    extr2 = extr.clone().requires_grad_()

    cov3d1.retain_grad()
    cov3d2.retain_grad()

    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        # run forward pytorch
        (
            out_conic_pytorch, 
            out_radius_pytorch, 
            out_tiles_pytorch
        ) = ewa_project_torch_impl(
            xyz1,
            cov3d1, 
            intr1,
            extr1,
            uv,
            W, H,
            visible
        )
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")
    
    t = time.time()
    for i in range(iters):
        (
            out_conic_cuda, 
            out_radius_cuda, 
            out_tiles_cuda
        ) = ms.ewa_project(
            xyz2,
            cov3d2, 
            intr2,
            extr2,
            uv,
            W, H,
            visible
        )
    
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(out_conic_pytorch, out_conic_cuda)
    torch.testing.assert_close(out_radius_pytorch, out_radius_cuda)
    torch.testing.assert_close(out_tiles_pytorch, out_tiles_cuda)
    print("Forward pass.")
    
    # ============================================ Backward =====================================
    print("backward: ")
    t = time.time()
    loss = out_conic_pytorch.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    loss2 = out_conic_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    cov3d1.grad = torch.nan_to_num(cov3d1.grad)
    xyz1.grad = torch.nan_to_num(xyz1.grad)
    if intr1.grad is not None:
        intr1.grad = torch.nan_to_num(intr1.grad)
    if extr1.grad is not None:
        extr1.grad = torch.nan_to_num(extr1.grad) 
    
    torch.testing.assert_close(cov3d1.grad, cov3d2.grad)
    torch.testing.assert_close(xyz1.grad, xyz2.grad)
    torch.testing.assert_close(intr1.grad, intr2.grad)
    torch.testing.assert_close(extr1.grad, extr2.grad)
    print("Backward pass.")
    