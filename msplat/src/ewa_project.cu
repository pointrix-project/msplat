/**
 * @file ewa_project.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <torch/torch.h>
#include <utils.h>


namespace cg = cooperative_groups;

__global__ void EWAProjectForwardCUDAKernel(int P,
                                            const float3 *xyz,
                                            const float *cov3d,
                                            const float *intr,
                                            const float *extr,
                                            const float2 *uv,
                                            const dim3 grid,
                                            const bool *visible,
                                            float3 *conic,
                                            int *radius,
                                            int *tiles) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visible[idx])
        return;

    float fx = intr[0];
    float fy = intr[1];

    float3 p = xyz[idx];
    float3 t = {
        extr[0] * p.x + extr[1] * p.y + extr[2] * p.z + extr[3],
        extr[4] * p.x + extr[5] * p.y + extr[6] * p.z + extr[7],
        extr[8] * p.x + extr[9] * p.y + extr[10] * p.z + extr[11]
    };

    glm::mat3 J = glm::mat3(fx / t.z, 0.0f, 0.0f,
                            0.0f, fy / t.z, 0.0f,
                            -(fx * t.x) / (t.z * t.z), -(fy * t.y) / (t.z * t.z), 0.0f);

    glm::mat3 W = glm::mat3(extr[0], extr[4], extr[8],
                            extr[1], extr[5], extr[9],
                            extr[2], extr[6], extr[10]);

    glm::mat3 T = J * W;

    glm::mat3 Vrk = glm::mat3(cov3d[6 * idx + 0], cov3d[6 * idx + 1], cov3d[6 * idx + 2],
                              cov3d[6 * idx + 1], cov3d[6 * idx + 3], cov3d[6 * idx + 4],
                              cov3d[6 * idx + 2], cov3d[6 * idx + 4], cov3d[6 * idx + 5]);

    glm::mat3 cov2D =  T * Vrk  * glm::transpose(T);

    float3 cov = {float(cov2D[0][0] + 0.3f),
                  float(cov2D[0][1]),
                  float(cov2D[1][1] + 0.3f)};

    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    uint2 rect_min, rect_max;

    get_rect(uv[idx], my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    float det_inv = 1.f / det;
    conic[idx].x = cov.z * det_inv;
    conic[idx].y = -cov.y * det_inv;
    conic[idx].z = cov.x * det_inv;

    radius[idx] = my_radius;
    tiles[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__global__ void EWAProjectBackCUDAKernel(int P,
                                         const float3 *xyz,
                                         const float *cov3d,
                                         const float *intr,
                                         const float *extr,
                                         const int *radius,
                                         const float3 *dL_dconic,
                                         float3 *dL_dxyz,
                                         float *dL_dcov3d,
                                         float *dL_dintr,
                                         float *dL_dextr) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radius[idx] > 0))
        return;

    float fx = intr[0];
    float fy = intr[1];

    float3 p = xyz[idx];
    const float *c3ptr = cov3d + 6 * idx;

    float3 t = {
        extr[0] * p.x + extr[1] * p.y + extr[2] * p.z + extr[3],
        extr[4] * p.x + extr[5] * p.y + extr[6] * p.z + extr[7],
        extr[8] * p.x + extr[9] * p.y + extr[10] * p.z + extr[11]
    };

    glm::mat3 J = glm::mat3(fx / t.z, 0.0f, 0.0f,
                            0.0f, fy / t.z, 0.0f,
                            -(fx * t.x) / (t.z * t.z), -(fy * t.y) / (t.z * t.z), 0.0f);

    glm::mat3 W = glm::mat3(extr[0], extr[4], extr[8],
                            extr[1], extr[5], extr[9],
                            extr[2], extr[6], extr[10]);

    glm::mat3 T = J * W;

    glm::mat3 Vrk = glm::mat3(c3ptr[0], c3ptr[1], c3ptr[2],
                              c3ptr[1], c3ptr[3], c3ptr[4],
                              c3ptr[2], c3ptr[4], c3ptr[5]);

    glm::mat3 cov2D =  T * Vrk  * glm::transpose(T);

    float3 cov = {float(cov2D[0][0] + 0.3f),
                  float(cov2D[0][1]),
                  float(cov2D[1][1] + 0.3f)};

    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;

    float nom = 1.0f / (det * det);

    float dL_dcovx = nom * (-cov.z * cov.z * dL_dconic[idx].x + cov.y * cov.z * dL_dconic[idx].y +
                         (det - cov.x * cov.z) * dL_dconic[idx].z);
    float dL_dcovy = nom * (2 * cov.y * cov.z * dL_dconic[idx].x -
                         (det + 2 * cov.y * cov.y) * dL_dconic[idx].y +
                         2 * cov.x * cov.y * dL_dconic[idx].z);
    float dL_dcovz = nom * ((det - cov.x * cov.z) * dL_dconic[idx].x +
                         cov.x * cov.y * dL_dconic[idx].y - cov.x * cov.x * dL_dconic[idx].z);
    
    dL_dcov3d[6 * idx + 0] += T[0][0] * T[0][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 0] += T[0][0] * T[0][1] * dL_dcovy;
    dL_dcov3d[6 * idx + 0] += T[0][1] * T[0][1] * dL_dcovz;

    dL_dcov3d[6 * idx + 1] += 2 * T[0][0] * T[1][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 1] += (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dcovy;
    dL_dcov3d[6 * idx + 1] += 2 * T[0][1] * T[1][1] * dL_dcovz;

    dL_dcov3d[6 * idx + 2] += 2 * T[0][0]* T[2][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 2] += (T[0][0]* T[2][1] + T[0][1]* T[2][0]) * dL_dcovy;
    dL_dcov3d[6 * idx + 2] += 2 * T[0][1]* T[2][1] * dL_dcovz;

    dL_dcov3d[6 * idx + 3] += T[1][0] * T[1][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 3] += T[1][0] * T[1][1] * dL_dcovy;
    dL_dcov3d[6 * idx + 3] += T[1][1] * T[1][1] * dL_dcovz;

    dL_dcov3d[6 * idx + 4] += 2 * T[1][0] * T[2][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 4] += (T[1][0] * T[2][1] + T[1][1] * T[2][0]) * dL_dcovy;
    dL_dcov3d[6 * idx + 4] += 2 * T[1][1] * T[2][1] * dL_dcovz;

    dL_dcov3d[6 * idx + 5] += T[2][0] * T[2][0] * dL_dcovx;
    dL_dcov3d[6 * idx + 5] += T[2][0] * T[2][1] * dL_dcovy;
    dL_dcov3d[6 * idx + 5] += T[2][1] * T[2][1] * dL_dcovz;

    float dL_dT00, dL_dT01, dL_dT10, dL_dT11, dL_dT20, dL_dT21;
    dL_dT00 = dL_dT01 = dL_dT10 = dL_dT11 = dL_dT20 = dL_dT21 = 0;

    dL_dT00 += 2 * (T[0][0] * c3ptr[0] + T[1][0] * c3ptr[1] + T[2][0] * c3ptr[2]) * dL_dcovx;
    dL_dT00 += (T[0][1] * c3ptr[0] + T[1][1] * c3ptr[1] + T[2][1] * c3ptr[2]) * dL_dcovy;

    dL_dT01 += (T[0][0] * c3ptr[0] + T[1][0] * c3ptr[1] + T[2][0] * c3ptr[2]) * dL_dcovy;
    dL_dT01 += 2 * (T[0][1] * c3ptr[0] + T[1][1] * c3ptr[1]  + T[2][1] * c3ptr[2]) * dL_dcovz;

    dL_dT10 += 2 * (T[0][0] * c3ptr[1] + T[1][0] * c3ptr[3] + T[2][0] * c3ptr[4]) * dL_dcovx;
    dL_dT10 += (T[0][1] * c3ptr[1] + T[1][1] * c3ptr[3] + T[2][1] * c3ptr[4]) * dL_dcovy;

    dL_dT11 += (T[0][0] * c3ptr[1] + T[1][0] * c3ptr[3] + T[2][0] * c3ptr[4]) * dL_dcovy;
    dL_dT11 += 2 * (T[0][1] * c3ptr[1] + T[1][1] * c3ptr[3] + T[2][1] * c3ptr[4]) * dL_dcovz;

    dL_dT20 += 2 * (T[0][0] * c3ptr[2] + T[1][0] * c3ptr[4] + T[2][0] * c3ptr[5]) * dL_dcovx;
    dL_dT20 += (T[0][1] * c3ptr[2] + T[1][1] * c3ptr[4] + T[2][1] * c3ptr[5]) * dL_dcovy;

    dL_dT21 += (T[0][0] * c3ptr[2] + T[1][0] * c3ptr[4] + T[2][0] * c3ptr[5]) * dL_dcovy;
    dL_dT21 += 2 * (T[0][1] * c3ptr[2] + T[1][1] * c3ptr[4] + T[2][1] * c3ptr[5]) * dL_dcovz;

    float dL_dJ00 = W[0][0] * dL_dT00 + W[1][0] * dL_dT10 + W[2][0] * dL_dT20;
    float dL_dJ20 = W[0][2] * dL_dT00 + W[1][2] * dL_dT10 + W[2][2] * dL_dT20;
    float dL_dJ11 = W[0][1] * dL_dT01 + W[1][1] * dL_dT11 + W[2][1] * dL_dT21;
    float dL_dJ21 = W[0][2] * dL_dT01 + W[1][2] * dL_dT11 + W[2][2] * dL_dT21;

    float tz = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    float dL_dtx = -fx * tz2 * dL_dJ20;
    float dL_dty = -fy * tz2 * dL_dJ21;
    float dL_dtz = -fx * tz2 * dL_dJ00 - fy * tz2 * dL_dJ11 +
                   (2 * fx * t.x) * tz3 * dL_dJ20 +
                   (2 * fy * t.y) * tz3 * dL_dJ21;
    
    if(dL_dintr != nullptr){
        atomicAdd(&dL_dintr[0], tz * dL_dJ00);
        atomicAdd(&dL_dintr[0], - t.x * tz2 * dL_dJ20);

        atomicAdd(&dL_dintr[1], tz * dL_dJ11);
        atomicAdd(&dL_dintr[1], - t.y * tz2 * dL_dJ21);
    }

    if(dL_dextr != nullptr){
        // dL/dT * dT/dextr
        // extr[0] extr[1] extr[2]         W[0][0] W[1][0] W[2][0]
        // extr[4] extr[5] extr[6]   ===>  W[0][1] W[1][1] W[2][1]
        // extr[8] extr[9] extr[10]        W[0][2] W[1][2] W[2][2]
        atomicAdd(&dL_dextr[0], J[0][0] * dL_dT00);
        atomicAdd(&dL_dextr[1], J[0][0] * dL_dT10);
        atomicAdd(&dL_dextr[2], J[0][0] * dL_dT20);

        atomicAdd(&dL_dextr[4], J[1][1] * dL_dT01);
        atomicAdd(&dL_dextr[5], J[1][1] * dL_dT11);
        atomicAdd(&dL_dextr[6], J[1][1] * dL_dT21);

        atomicAdd(&dL_dextr[8], J[2][0] * dL_dT00 + J[2][1] * dL_dT01);
        atomicAdd(&dL_dextr[9], J[2][0] * dL_dT10 + J[2][1] * dL_dT11);     
        atomicAdd(&dL_dextr[10], J[2][0] * dL_dT20 + J[2][1] * dL_dT21);

        // dL/dt * dt/dextr
        atomicAdd(&dL_dextr[0], p.x * dL_dtx); //
        atomicAdd(&dL_dextr[1], p.y * dL_dtx); 
        atomicAdd(&dL_dextr[2], p.z * dL_dtx); 
        atomicAdd(&dL_dextr[3], dL_dtx); //

        atomicAdd(&dL_dextr[4], p.x * dL_dty); 
        atomicAdd(&dL_dextr[5], p.y * dL_dty); //
        atomicAdd(&dL_dextr[6], p.z * dL_dty); 
        atomicAdd(&dL_dextr[7], dL_dty); //

        atomicAdd(&dL_dextr[8], p.x * dL_dtz); 
        atomicAdd(&dL_dextr[9], p.y * dL_dtz);
        atomicAdd(&dL_dextr[10], p.z * dL_dtz); // 
        atomicAdd(&dL_dextr[11], dL_dtz); //
    }

    dL_dxyz[idx] = {
        extr[0] * dL_dtx + extr[4] * dL_dty + extr[8] * dL_dtz,
        extr[1] * dL_dtx + extr[5] * dL_dty + extr[9] * dL_dtz,
        extr[2] * dL_dtx + extr[6] * dL_dty + extr[10] * dL_dtz};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(const torch::Tensor &xyz,
                  const torch::Tensor &cov3d,
                  const torch::Tensor &intr,
                  const torch::Tensor &extr,
                  const torch::Tensor &uv,
                  const int W,
                  const int H,
                  const torch::Tensor &visible) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(intr);
    CHECK_INPUT(extr);
    CHECK_INPUT(uv);
    CHECK_INPUT(visible);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    auto int_opts = xyz.options().dtype(torch::kInt32);

    torch::Tensor conic = torch::zeros({P, 3}, float_opts);
    torch::Tensor radius = torch::zeros({P}, int_opts);
    torch::Tensor tiles = torch::zeros({P}, int_opts);

    if (P != 0) {
        const dim3 tile_grid(
            (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

        EWAProjectForwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float3 *)xyz.contiguous().data_ptr<float>(),
            cov3d.contiguous().data_ptr<float>(),
            intr.contiguous().data_ptr<float>(),
            extr.contiguous().data_ptr<float>(),
            (float2 *)uv.contiguous().data_ptr<float>(),
            tile_grid,
            visible.contiguous().data_ptr<bool>(),
            (float3 *)conic.data_ptr<float>(),
            radius.data_ptr<int>(),
            tiles.data_ptr<int>());
    }

    return std::make_tuple(conic, radius, tiles);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectBackward(const torch::Tensor &xyz,
                   const torch::Tensor &cov3d,
                   const torch::Tensor &intr,
                   const torch::Tensor &extr,
                   const torch::Tensor &radius,
                   const torch::Tensor &dL_dconic) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(intr);
    CHECK_INPUT(extr);
    CHECK_INPUT(radius);
    CHECK_INPUT(dL_dconic);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dcov3d = torch::zeros({P, 6}, float_opts);
    torch::Tensor dL_dintr = torch::zeros({4}, float_opts);
    torch::Tensor dL_dextr = torch::zeros({3, 4}, float_opts);

    float *dL_dintr_ptr = nullptr;
    if (intr.requires_grad())
        dL_dintr_ptr = dL_dintr.data_ptr<float>();

    float *dL_dextr_ptr = nullptr;
    if (extr.requires_grad())
        dL_dextr_ptr = dL_dextr.data_ptr<float>();

    if (P != 0) {
        // Kernel
        EWAProjectBackCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float3 *)xyz.contiguous().data_ptr<float>(),
            cov3d.contiguous().data_ptr<float>(),
            intr.contiguous().data_ptr<float>(),
            extr.contiguous().data_ptr<float>(),
            radius.contiguous().data_ptr<int>(),
            (float3 *)dL_dconic.contiguous().data_ptr<float>(),
            (float3 *)dL_dxyz.data_ptr<float>(),
            dL_dcov3d.data_ptr<float>(), 
            dL_dintr_ptr,
            dL_dextr_ptr);
    }

    return std::make_tuple(dL_dxyz, dL_dcov3d, dL_dintr, dL_dextr);
}