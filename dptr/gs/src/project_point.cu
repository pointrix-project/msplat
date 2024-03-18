/**
 * @file project_point_kernel.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <utils.h>

namespace cg = cooperative_groups;

__global__ void projectPointForwardCUDAKernel(const int P,
                                              const float3 *xyz,
                                              const float *intr,
                                              const float *extr,
                                              const int W,
                                              const int H,
                                              float2 *uv,
                                              float *depth,
                                              const float nearest,
                                              const float extent) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p = xyz[idx];
    float3 tmp = {extr[0] * p.x + extr[1] * p.y + extr[2] * p.z + extr[3],
                  extr[4] * p.x + extr[5] * p.y + extr[6] * p.z + extr[7],
                  extr[8] * p.x + extr[9] * p.y + extr[10] * p.z + extr[11]};
    float norm1 = 1.0 / (tmp.z + 1e-7);

    // to pixel coordinate
    float3 uvd = {intr[0] * tmp.x * norm1 + intr[2] - 0.5,
                  intr[1] * tmp.y * norm1 + intr[3] - 0.5,
                  tmp.z};

    // perform culling
    bool near_culling = false;
    if (nearest > 0)
        near_culling = uvd.z <= nearest;

    bool extent_culling = false;
    if (extent > 0) {
        float x_limit_min = (1 - extent) * W * 0.5;
        float x_limit_max = (1 + extent) * W * 0.5;
        float y_limit_min = (1 - extent) * H * 0.5;
        float y_limit_max = (1 + extent) * H * 0.5;
        extent_culling = uvd.x < x_limit_min || uvd.x > x_limit_max ||
                         uvd.y < y_limit_min || uvd.y > y_limit_max;
    }
    if (near_culling || extent_culling)
        return;

    uv[idx] = {uvd.x, uvd.y};
    depth[idx] = uvd.z;
}

__global__ void projectPointBackwardCUDAKernel(const int P,
                                               const float3 *xyz,
                                               const float *intr,
                                               const float *extr,
                                               const int W,
                                               const int H,
                                               const float2 *uv,
                                               const float *depth,
                                               const float2 *dL_duv,
                                               const float *dL_ddepth,
                                               float3 *dL_dxyz,
                                               float *dL_dintr,
                                               float *dL_dextr) {
    auto idx = cg::this_grid().thread_rank();

    // depth == 0 means culled
    if (idx >= P || depth[idx] == 0)
        return;

    float3 p = xyz[idx];
    float3 tmp = {extr[0] * p.x + extr[1] * p.y + extr[2] * p.z + extr[3],
                  extr[4] * p.x + extr[5] * p.y + extr[6] * p.z + extr[7],
                  extr[8] * p.x + extr[9] * p.y + extr[10] * p.z + extr[11]};
    
    // tmp.z is depth[idx] != 0
    float norm1 = 1.0 / tmp.z;
    float norm2 = 1.0 / (tmp.z * tmp.z);

    // dL_dxyz
    dL_dxyz[idx].x +=
        (intr[0] * (extr[0] * tmp.z - tmp.x * extr[8]) * norm2) * dL_duv[idx].x;
    dL_dxyz[idx].x +=
        (intr[1] * (extr[4] * tmp.z - tmp.y * extr[8]) * norm2) * dL_duv[idx].y;
    dL_dxyz[idx].x += extr[8] * dL_ddepth[idx];

    dL_dxyz[idx].y +=
        (intr[0] * (extr[1] * tmp.z - tmp.x * extr[9]) * norm2) * dL_duv[idx].x;
    dL_dxyz[idx].y +=
        (intr[1] * (extr[5] * tmp.z - tmp.y * extr[9]) * norm2) * dL_duv[idx].y;
    dL_dxyz[idx].y += extr[9] * dL_ddepth[idx];

    dL_dxyz[idx].z += (intr[0] * (extr[2] * tmp.z - tmp.x * extr[10]) * norm2) *
                      dL_duv[idx].x;
    dL_dxyz[idx].z += (intr[1] * (extr[6] * tmp.z - tmp.y * extr[10]) * norm2) *
                      dL_duv[idx].y;
    dL_dxyz[idx].z += extr[10] * dL_ddepth[idx];

    // dL_dintr
    if (dL_dintr != nullptr) {
        atomicAdd(&dL_dintr[0], tmp.x * norm1 * dL_duv[idx].x);
        atomicAdd(&dL_dintr[1], tmp.y * norm1 * dL_duv[idx].y);
        atomicAdd(&dL_dintr[2], dL_duv[idx].x);
        atomicAdd(&dL_dintr[3], dL_duv[idx].y);
    }

    // dL_dextr
    if (dL_dextr != nullptr) {
        atomicAdd(&dL_dextr[0], intr[0] * p.x * norm1 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[1], intr[0] * p.y * norm1 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[2], intr[0] * p.z * norm1 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[3], intr[0] * norm1 * dL_duv[idx].x);

        atomicAdd(&dL_dextr[4], intr[1] * p.x * norm1 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[5], intr[1] * p.y * norm1 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[6], intr[1] * p.z * norm1 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[7], intr[1] * norm1 * dL_duv[idx].y);

        // may be bugs?
        atomicAdd(&dL_dextr[8], -intr[0] * p.x * tmp.x * norm2 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[8], -intr[1] * p.x * tmp.y * norm2 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[8], p.x * dL_ddepth[idx]);

        atomicAdd(&dL_dextr[9], -intr[0] * p.y * tmp.x * norm2 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[9], -intr[1] * p.y * tmp.y * norm2 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[9], p.y * dL_ddepth[idx]);

        atomicAdd(&dL_dextr[10],
                  -intr[0] * p.z * tmp.x * norm2 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[10],
                  -intr[1] * p.z * tmp.y * norm2 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[10], p.z * dL_ddepth[idx]);

        atomicAdd(&dL_dextr[11], -intr[0] * tmp.x * norm2 * dL_duv[idx].x);
        atomicAdd(&dL_dextr[11], -intr[1] * tmp.y * norm2 * dL_duv[idx].y);
        atomicAdd(&dL_dextr[11], dL_ddepth[idx]);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
projectPointsForward(const torch::Tensor &xyz,
                     const torch::Tensor &intr,
                     const torch::Tensor &extr,
                     const int W,
                     const int H,
                     const float nearest,
                     const float extent) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(intr);
    CHECK_INPUT(extr);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor uv = torch::zeros({P, 2}, float_opts);
    torch::Tensor depth = torch::zeros({P, 1}, float_opts);

    if (P != 0) {
        projectPointForwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float3 *)xyz.contiguous().data_ptr<float>(),
            intr.contiguous().data_ptr<float>(),
            extr.contiguous().data_ptr<float>(),
            W,
            H,
            (float2 *)uv.data_ptr<float>(),
            depth.data_ptr<float>(),
            nearest,
            extent);
    }

    return std::make_tuple(uv, depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectPointsBackward(const torch::Tensor &xyz,
                      const torch::Tensor &intr,
                      const torch::Tensor &extr,
                      const int W,
                      const int H,
                      const torch::Tensor &uv,
                      const torch::Tensor &depth,
                      const torch::Tensor &dL_duv,
                      const torch::Tensor &dL_ddepth) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(intr);
    CHECK_INPUT(extr);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    // auto double_opts = xyz.options().dtype(torch::kFloat64);
    torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dintr = torch::zeros({4}, float_opts);
    torch::Tensor dL_dextr = torch::zeros({3, 4}, float_opts);

    float *dL_dintr_ptr = nullptr;
    if (intr.requires_grad())
        dL_dintr_ptr = dL_dintr.data_ptr<float>();

    float *dL_dextr_ptr = nullptr;
    if (extr.requires_grad())
        dL_dextr_ptr = dL_dextr.data_ptr<float>();

    if (P != 0) {
        projectPointBackwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float3 *)xyz.contiguous().data_ptr<float>(),
            intr.contiguous().data_ptr<float>(),
            extr.contiguous().data_ptr<float>(),
            W,
            H,
            (float2 *)uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            (float2 *)dL_duv.contiguous().data_ptr<float>(),
            dL_ddepth.contiguous().data_ptr<float>(),
            (float3 *)dL_dxyz.data_ptr<float>(),
            dL_dintr_ptr,
            dL_dextr_ptr);
    }

    return std::make_tuple(dL_dxyz, dL_dintr, dL_dextr);
}