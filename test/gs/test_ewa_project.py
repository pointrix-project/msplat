
import time
import math
import torch
import DifferentiablePointRender.GaussianSplatting as gs

BLOCK_X = 16
BLOCK_Y = 16

def ewa_project_torch_impl(
    cov3d, 
    viewmat,
    camparam,
    uv,
    depth,
    xy,
    W, H
):    
    # return cov2d, radii, tiles_touched
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape
    
    viewmat = viewmat.transpose(-2, -1)
    
    Wmat = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    
    # take care of grad
    t = torch.cat([uv[..., 0:1] * depth[..., 0:1], 
                   uv[..., 1:2] * depth[..., 0:1], 
                   depth[..., 0:1]], dim=-1)
    
    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz**2  # (...,)
    
    fx = camparam[..., 0]
    fy = camparam[..., 1]
    
    Jmat = torch.stack(
        [
            torch.stack([fx * rz, torch.zeros_like(rz), -fx * t[..., 0] * rz2], dim=-1),
            torch.stack([torch.zeros_like(rz), fy * rz, -fy * t[..., 1] * rz2], dim=-1),
        ],
        dim=-2,
    )  # (..., 2, 3)
    
    T = Jmat @ Wmat  # (..., 2, 3)
    cov3d_1 = torch.stack([cov3d[..., 0], cov3d[..., 1], cov3d[..., 2]], dim=-1)
    cov3d_2 = torch.stack([cov3d[..., 1], cov3d[..., 3], cov3d[..., 4]], dim=-1)
    cov3d_3 = torch.stack([cov3d[..., 2], cov3d[..., 4], cov3d[..., 5]], dim=-1)
    cov3d = torch.stack([cov3d_1, cov3d_2, cov3d_3], dim=-1)
    
    cov2d = T @ cov3d @ T.transpose(-1, -2)  # (..., 2, 2)
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + 0.3
    
    # Compute extent in screen space
    det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2
    det_mask = det != 0
    det = torch.clamp(det, min=1e-8)
        
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # (..., 3)
    
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    
    # get tiles_touched
    top_left = ((xy - radius[..., None])/BLOCK_X).to(torch.int32)
    bottom_right = ((xy + radius[..., None] + BLOCK_Y - 1)/BLOCK_Y).to(torch.int32)
    
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
    tiles_touched = tiles_tmp[..., 0] * tiles_tmp[..., 1]
    
    mask = torch.logical_and(tiles_touched != 0, det_mask)
    
    conic = conic * mask.float()[..., None]
    radius = radius * mask.float()
    tiles_touched = tiles_touched * mask.float()
    
    return conic, radius.int(), tiles_touched.int()


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
    seed = 3
    torch.manual_seed(seed)
    torch.set_printoptions(precision=10)
    
    iters = 100
    N = 1
    
    print("=============================== running test on ewa_project ===============================")
    
    W = 800
    H = 800
    
    fovx = 0.6911112070083618
    fovy = 0.6911112070083618
    
    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)
    
    viewmat = torch.Tensor([[6.1182e-01,  7.9096e-01, -6.7348e-03,  0.0000e+00], 
                            [ 7.9099e-01, -6.1180e-01,  5.2093e-03,  0.0000e+00], 
                            [ 1.3906e-14, -8.5126e-03, -9.9996e-01,  0.0000e+00], 
                            [ 1.1327e-09,  1.0458e-09,  4.0311e+00,  1.0000e+00]]).cuda()
    projmat = getProjectionMatrix(fovx, fovy).transpose(0, 1)
    full_proj_transform = (viewmat.unsqueeze(0).bmm(projmat.unsqueeze(0))).squeeze(0)
    
    camparam = torch.Tensor([fx, fy, H/2, W/2]).cuda()
    xyz = torch.randn((N, 3)).cuda() * 2.6 - 1.3
    
    rand_scale = torch.randn(N, 3, device="cuda", dtype=torch.float)
    rand_quats = torch.randn(N, 4, device="cuda", dtype=torch.float)
    rand_uquats = rand_quats / torch.norm(rand_quats, 2, dim=-1, keepdim=True)
    
    cov3d = gs.compute_cov3d(rand_scale, rand_uquats)
    
    # project points
    (
        uv,
        depth 
    ) = gs.project_point(
        xyz, 
        viewmat, 
        full_proj_transform,
        camparam,
        W, H)

    visibility_status = (depth != 0).squeeze(-1)
    
    xy = uv.clone()
    xy[:, 0] = ndc_to_pixel(uv[:, 0], W)
    xy[:, 1] = ndc_to_pixel(uv[:, 1], H)
    
    # mask out, otherwise there will be nan in the grad of torch_impl. 
    cov3d1 = cov3d.clone()[visibility_status].requires_grad_()
    cov3d2 = cov3d.clone()[visibility_status].requires_grad_()
    uv = uv[visibility_status]
    depth1 = depth.clone()[visibility_status].requires_grad_()
    depth2 = depth.clone()[visibility_status].requires_grad_()
    xy = xy[visibility_status]
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        # run forward pytorch
        (
            out_conic_pytorch, 
            out_radius_pytorch, 
            out_tiles_touched_pytorch
        ) = ewa_project_torch_impl(
            cov3d1, 
            viewmat,
            camparam,
            uv,
            depth1,
            xy,
            W, H
        )
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")
    
    t = time.time()
    for i in range(iters):
        (
            out_conic_cuda, 
            out_radius_cuda, 
            out_tiles_touched_cuda
        ) = gs.ewa_project(
            cov3d2, 
            viewmat,
            camparam,
            uv,
            depth2,
            xy,
            W, H
        )
    
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    print(out_tiles_touched_pytorch)
    exit(-1)
    
    torch.testing.assert_close(out_conic_pytorch, out_conic_cuda, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_radius_pytorch, out_radius_cuda)
    torch.testing.assert_close(out_tiles_touched_pytorch, out_tiles_touched_cuda)
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
    
    # print(depth1.grad)
    # print(depth2.grad)
    
    print(cov3d1.grad)
    print(cov3d2.grad)
    torch.testing.assert_close(cov3d1.grad, cov3d2.grad)
    # torch.testing.assert_close(depth1.grad, depth2.grad)
    print("Backward pass.")
    