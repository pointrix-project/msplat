
import time
import math
import torch
import DifferentiablePointRender.GaussianSplatting as gs


def ndc_to_pixel(ndc, size):
    return ((ndc + 1.0) * size - 1.0) * 0.5


def project_point_torch_impl(
    xyz, 
    viewmat, 
    projmat,
    camparam,
    W,
    H,
    nearest=0.2,
    extent=1.3
):
    viewmat = viewmat.transpose(0, 1)
    projmat = projmat.transpose(0, 1)
    
    tmp_one = torch.ones_like(xyz[:, 0:1])
    xyz_hom = torch.cat([xyz, tmp_one], dim=-1)
    
    p_hom = torch.bmm(projmat.unsqueeze(0).repeat(xyz.shape[0], 1, 1), xyz_hom.unsqueeze(-1)).squeeze(-1)
    p_w = 1.0 / (p_hom[:, -1].unsqueeze(-1) + 1e-7)
    p_proj = p_hom * p_w
    
    p_view = torch.bmm(viewmat.unsqueeze(0).repeat(xyz.shape[0], 1, 1), xyz_hom.unsqueeze(-1)).squeeze(-1)
    
    uv = torch.stack([p_proj[:, 0], p_proj[:, 1]], dim=-1)
    depth = p_view[:, -2]
    
    near_mask = depth <= nearest
    extent_mask_x = torch.logical_or(p_proj[:, 0] < -extent, p_proj[:, 0] > extent)
    extent_mask_y = torch.logical_or(p_proj[:, 1] < -extent, p_proj[:, 1] > extent)
    extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
    mask = torch.logical_or(near_mask, extent_mask)
    
    uv[:, 0][mask] = 0
    uv[:, 1][mask] = 0
    depth[mask] = 0
    
    return uv, depth.unsqueeze(-1)


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

if __name__ == "__main__":
    # seed = 123
    # torch.manual_seed(seed)
    # torch.set_printoptions(precision=10)
    
    iters = 1
    N = 100000
    
    print("=============================== running test on project_points ===============================")
    
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
    
    xyz1 = xyz.clone().requires_grad_()
    xyz2 = xyz.clone().requires_grad_()
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        (
            out_pytorch_uv, 
            out_pytorch_depth
        ) = project_point_torch_impl(
            xyz1, 
            viewmat, 
            full_proj_transform,
            camparam,
            W, H)
        
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")
    
    t = time.time()
    for i in range(iters):
        (
            out_cuda_uv,
            out_cuda_depth 
        ) = gs.project_point(
            xyz2, 
            viewmat, 
            full_proj_transform,
            camparam,
            W, H)
    
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(out_pytorch_uv, out_cuda_uv)
    torch.testing.assert_close(out_pytorch_depth, out_cuda_depth)
    print("Forward pass.")
    
    # ============================================ Backward =====================================
    print("backward: ")
    t = time.time()
    loss = out_pytorch_uv.sum() + out_pytorch_depth.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    loss2 = out_cuda_uv.sum() + out_cuda_depth.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(xyz1.grad, xyz2.grad)
    print("Backward pass.")