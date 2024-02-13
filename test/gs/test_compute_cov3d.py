
import time
import torch
import dptr.gs as gs


def compute_cov3d_torch_impl(scales, uquats):
    # Scale vector [N, 3] to matrix S[N, 3, 3]
    S = torch.diag_embed(scales)
    
    # Quaternion [N, 4] to rotation matrix R[N, 3, 3]
    batch_size = uquats.size(0)
    R = torch.zeros((batch_size, 3, 3), dtype=scales.dtype, device=scales.device)
    
    R[:, 0, 0] = 1 - 2 * (uquats[:, 2]**2 + uquats[:, 3]**2)
    R[:, 0, 1] = 2 * (uquats[:, 1] * uquats[:, 2] - uquats[:, 0] * uquats[:, 3])
    R[:, 0, 2] = 2 * (uquats[:, 1] * uquats[:, 3] + uquats[:, 0] * uquats[:, 2])

    R[:, 1, 0] = 2 * (uquats[:, 1] * uquats[:, 2] + uquats[:, 0] * uquats[:, 3])
    R[:, 1, 1] = 1 - 2 * (uquats[:, 1]**2 + uquats[:, 3]**2)
    R[:, 1, 2] = 2 * (uquats[:, 2] * uquats[:, 3] - uquats[:, 0] * uquats[:, 1])

    R[:, 2, 0] = 2 * (uquats[:, 1] * uquats[:, 3] - uquats[:, 0] * uquats[:, 2])
    R[:, 2, 1] = 2 * (uquats[:, 2] * uquats[:, 3] + uquats[:, 0] * uquats[:, 1])
    R[:, 2, 2] = 1 - 2 * (uquats[:, 1]**2 + uquats[:, 2]**2)
    
    # Sigma = SR
    Sigma = torch.matmul(R, S)

    # Cov3d = RS(SR)T
    cov3d = torch.matmul(Sigma, Sigma.transpose(1, 2))

    # Output the 6 parameters in the up-right corner as a vector
    output_vector = torch.stack([cov3d[:, 0, 0], cov3d[:, 0, 1], cov3d[:, 0, 2], cov3d[:, 1, 1], cov3d[:, 1, 2], cov3d[:, 2, 2]], dim=-1)

    return output_vector

if __name__ == "__main__":
    iters = 1000
    N = 2000
    
    print("=============================== running test on compute_cov3d ===============================")
    # generate data
    rand_scale = torch.rand(N, 3, device="cuda", dtype=torch.float)
    rand_quats = torch.randn(N, 4, device="cuda", dtype=torch.float)
    rand_uquats = rand_quats / torch.norm(rand_quats, 2, dim=-1, keepdim=True)
    
    scales1 = rand_scale.clone().requires_grad_()
    scales2 = rand_scale.clone().requires_grad_()
    
    uquats1 = rand_uquats.clone().requires_grad_()
    uquats2 = rand_uquats.clone().requires_grad_()
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        out_pytorch = compute_cov3d_torch_impl(scales1, uquats1)
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    for i in range(iters):
        out_cuda = gs.compute_cov3d(scales2, uquats2)
    
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    torch.testing.assert_close(out_pytorch, out_cuda)
    print("Forward pass.")
    
    # ============================================ Backward =====================================
    print("backward: ")
    t = time.time()
    loss = out_pytorch.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    loss2 = out_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(scales1.grad, scales2.grad)
    torch.testing.assert_close(uquats1.grad, uquats2.grad)
    print("Backward pass.")