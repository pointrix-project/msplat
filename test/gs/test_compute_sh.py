
import time
import torch
import dptr.gs as gs


def compute_sh_torch_impl(sh_coeffs, viewdirs):
    *dims, dim_sh, C = sh_coeffs.shape
    
    bases = eval_sh_bases(dim_sh, viewdirs)
    
    return torch.clamp((bases[..., None] * sh_coeffs).sum(dim=-2) + 0.5, min=0.0)

"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (
                        xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
                    
    return result


if __name__ == "__main__":
    
    iters = 1000
    N = 1000
    degree = 3
    
    print("=============================== running test on compute_sh ===============================")
    # generate data
    degree_dim = pow(degree+1, 2)
    
    viewdirs = torch.randn(N, 3).cuda()
    viewdirs /= torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    sh_coeffs = torch.randn(N, degree_dim, 3).cuda()
    
    sh_coeffs1 = sh_coeffs.clone().requires_grad_()
    sh_coeffs2 = sh_coeffs.clone().requires_grad_()
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        out_pytorch = compute_sh_torch_impl(sh_coeffs1, viewdirs)
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    for i in range(iters):
        out_cuda = gs.compute_sh(sh_coeffs2, degree, viewdirs)
    
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
    
    torch.testing.assert_close(sh_coeffs1.grad, sh_coeffs2.grad)
    print("Backward pass.")
