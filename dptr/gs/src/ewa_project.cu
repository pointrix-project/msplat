
#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <glm/gtc/type_ptr.hpp>

namespace cg = cooperative_groups;

// __device__ inline void PrintMatrix(const glm::vec3 vec)
// {
//     float* data = (float*)glm::value_ptr(vec);
//     printf("[%f %f %f]\n", data[0], data[1], data[2]);
// }

// __device__ inline void PrintMatrix(const glm::mat3 mat)
// {   // Column Major
//     float* data = (float*)glm::value_ptr(mat);
//     printf("[");
//     for(int i = 0; i < 3; i++){
//         printf("[");
//         for(int j = 0; j < 3; j++){
//             printf("%f ", data[j * 3 + i]);
//         }
//         printf("]");
//         if(i < 2)
//             printf("\n");
//     }
//     printf("]\n");
// }


__global__ void EWAProjectForwardCUDAKernel(
    int P,
    const float3* xyz,
    const float* cov3d,
    const float* viewmat,
    const float* camparam,
    const float2* uv,
    const dim3 grid,
    const bool* visibility_status,
    float3* conic,
    int* radii,
    int* tiles_touched
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visibility_status[idx])
        return;

    float fx = camparam[0];
    float fy = camparam[1];

    float3 t = transform_point_4x3(viewmat, xyz[idx]);

    // const float limx = 1.3f * 0.36000002589322094;
	// const float limy = 1.3f * 0.36000002589322094;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        fx / t.z, 0.0f, -(fx * t.x) / (t.z * t.z),
        0.0f, fy / t.z, -(fy * t.y) / (t.z * t.z),
        0, 0, 0);
    
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3d[6 * idx + 0], cov3d[6 * idx + 1], cov3d[6 * idx + 2],
        cov3d[6 * idx + 1], cov3d[6 * idx + 3], cov3d[6 * idx + 4],
        cov3d[6 * idx + 2], cov3d[6 * idx + 4], cov3d[6 * idx + 5]);
    
    glm::mat3 ewa_cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    float3 cov = { 
        float(ewa_cov[0][0] + 0.3f), 
        float(ewa_cov[0][1]), 
        float(ewa_cov[1][1] + 0.3f)
    };
    
    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;

    // Compute extent in screen space
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    uint2 rect_min, rect_max;

    get_rect(uv[idx], my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // cov2d 
    float det_inv = 1.f / det;
    conic[idx].x = cov.z * det_inv;
    conic[idx].y = -cov.y * det_inv;
    conic[idx].z = cov.x * det_inv;

    // radius
    radii[idx] = my_radius;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


__global__ void EWAProjectBackCUDAKernel(
    int P,
    const float3* xyz,
	const float* cov3d,
	const float* viewmat,
	const float* camparam,
    const int* radii,
	const float3* dL_dconic,
	float3* dL_dxyz,
	float* dL_dcov3d
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    float fx = camparam[0];
    float fy = camparam[1];

    float3 t = transform_point_4x3(viewmat, xyz[idx]);

    glm::mat3 J = glm::mat3(
        fx / t.z, 0.0f, -(fx * t.x) / (t.z * t.z),
        0.0f, fy / t.z, -(fy * t.y) / (t.z * t.z),
        0, 0, 0
    );

    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    glm::mat3 Vrk = glm::mat3(
        cov3d[6*idx + 0], cov3d[6*idx + 1], cov3d[6*idx + 2],
        cov3d[6*idx + 1], cov3d[6*idx + 3], cov3d[6*idx + 4],
        cov3d[6*idx + 2], cov3d[6*idx + 4], cov3d[6*idx + 5]);

    glm::mat3 T = W * J;

    glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Use helper variables for 2D covariance entries. More compact.
    float a = cov2D[0][0] + 0.3f;
    float b = cov2D[0][1];
    float c = cov2D[1][1] + 0.3f;

    float det = a * c - b * b;
    if (det == 0)
        return;
       
    float nom = 1.0f / (det * det);

    float dL_da = nom * (- c * c * dL_dconic[idx].x + b * c * dL_dconic[idx].y + (det - a * c) * dL_dconic[idx].z);
    float dL_dc = nom * ((det - a * c) * dL_dconic[idx].x + a * b * dL_dconic[idx].y  - a * a * dL_dconic[idx].z);
    float dL_db = nom * (2 * b * c * dL_dconic[idx].x - (det + 2 * b * b) * dL_dconic[idx].y + 2 * a * b * dL_dconic[idx].z);

    dL_dcov3d[6 * idx + 0] += (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
    dL_dcov3d[6 * idx + 3] += (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
    dL_dcov3d[6 * idx + 5] += (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

    dL_dcov3d[6 * idx + 1] += 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
    dL_dcov3d[6 * idx + 2] += 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
    dL_dcov3d[6 * idx + 4] += 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
    
    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
        (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
    float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
        (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
    float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
        (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
    float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
        (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
    float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
        (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
    float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
        (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

    float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	float dL_dtx = -fx * tz2 * dL_dJ02;
	float dL_dty = -fy * tz2 * dL_dJ12;
	float dL_dtz = -fx * tz2 * dL_dJ00  - fy * tz2 * dL_dJ11 + (2 * fx * t.x) * tz3 * dL_dJ02 + (2 * fy * t.y) * tz3 * dL_dJ12;

	dL_dxyz[idx] = {
        viewmat[0] * dL_dtx + viewmat[1] * dL_dty + viewmat[2] * dL_dtz,
        viewmat[4] * dL_dtx + viewmat[5] * dL_dty + viewmat[6] * dL_dtz,
        viewmat[8] * dL_dtx + viewmat[9] * dL_dty + viewmat[10] * dL_dtz
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const int W, const int H,
    const torch::Tensor &visibility_status
){
    CHECK_INPUT(xyz);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(visibility_status);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    auto int_opts = xyz.options().dtype(torch::kInt32);

    torch::Tensor conic = torch::zeros({P, 3}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);

    if(P != 0){
        const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

        EWAProjectForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
            P, 
            (float3*)xyz.contiguous().data_ptr<float>(),
            cov3d.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            (float2*)uv.contiguous().data_ptr<float>(),
            tile_grid,
            visibility_status.contiguous().data_ptr<bool>(),
            (float3*)conic.data_ptr<float>(),
            radii.data_ptr<int>(),
            tiles_touched.data_ptr<int>()
        );
    }

    return std::make_tuple(conic, radii, tiles_touched);
}


std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& radii,
    const torch::Tensor& dL_dconic
){
    CHECK_INPUT(xyz);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(radii);
    CHECK_INPUT(dL_dconic);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dcov3d = torch::zeros({P, 6}, float_opts);
    
    if(P != 0){
        // Kernel
        EWAProjectBackCUDAKernel<<<(P + 255) / 256, 256 >>>(
            P, 
            (float3*)xyz.contiguous().data_ptr<float>(), 
            cov3d.contiguous().data_ptr<float>(), 
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            (float3*)dL_dconic.contiguous().data_ptr<float>(),
            (float3*)dL_dxyz.data_ptr<float>(),
            dL_dcov3d.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dxyz, dL_dcov3d);
}