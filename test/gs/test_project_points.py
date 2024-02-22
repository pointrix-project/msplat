
import time
import math
import torch
import dptr.gs as gs


def project_point_torch_impl(
    xyz, 
    intr, 
    extr,
    W,
    H,
    nearest=0.2,
    extent=1.3
):
    
    K = torch.eye(3).cuda()
    K[0, 0] = intr[0]
    K[1, 1] = intr[1]
    K[0, 2] = intr[2]
    K[1, 2] = intr[3]

    R = extr[:3, :3]
    t = extr[:3, -1].unsqueeze(dim=1)
    
    pt_cam = torch.matmul(R, xyz.t()) + t
    
    # Apply camera intrinsic matrix
    p_proj = torch.matmul(K, pt_cam)

    depth = p_proj[2]
    uv = p_proj[:2] / (depth + 1e-7) - 0.5

    uv = uv.t()
    
    near_mask = depth <= nearest
    extent_mask_x = torch.logical_or(uv[:, 0] < -extent * W * 0.5, uv[:, 0] > extent * W * 0.5)
    extent_mask_y = torch.logical_or(uv[:, 1] < -extent * H * 0.5, uv[:, 1] > extent * H * 0.5)
    extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
    mask = torch.logical_or(near_mask, extent_mask)
    
    uv_masked = uv.clone()
    depth_masked = depth.clone()
    uv_masked[:, 0][mask] = 0
    uv_masked[:, 1][mask] = 0
    depth_masked[mask] = 0
    
    return uv_masked, depth_masked.unsqueeze(-1)

if __name__ == "__main__":
    seed = 124
    torch.manual_seed(seed)
    
    iters = 10
    N = 100
    
    print("=============================== running test on project_points ===============================")
    
    W = 1600
    H = 1200
    
    intr = torch.Tensor([2892.33, 2883.18, 823.205, 619.071]).cuda().float()
    extr = torch.Tensor([
            [0.970263, 0.00747983, 0.241939, -191.02],
            [-0.0147429, 0.999493, 0.0282234, 3.2883],
            [-0.241605, -0.030951, 0.969881, 22.5401]
        ]).cuda().float()

    xyz = torch.rand((N, 3)).cuda()
    xyz[:, 0] = xyz[:, 0] * 500
    xyz[:, 1] = xyz[:, 1] * 500
    xyz[:, 2] = xyz[:, 2] * 400 + 400

    xyz1 = xyz.clone().requires_grad_()
    xyz2 = xyz.clone().requires_grad_()
    intr1 = intr.clone().requires_grad_()
    intr2 = intr.clone().requires_grad_()
    extr1 = extr.clone().requires_grad_()
    extr2 = extr.clone().requires_grad_()
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        (
            out_pytorch_uv, 
            out_pytorch_depth
        ) = project_point_torch_impl(
            xyz1, 
            intr1, 
            extr1,
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
            intr2, 
            extr2,
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

    print(extr1.grad)
    print(extr2.grad)
    print(extr1.grad - extr2.grad)
    torch.testing.assert_close(xyz1.grad, xyz2.grad)
    torch.testing.assert_close(intr1.grad, intr2.grad)
    torch.testing.assert_close(extr1.grad, extr2.grad)
    print("Backward pass.")