
#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <glm/gtc/type_ptr.hpp>

namespace cg = cooperative_groups;

__device__ inline void PrintMatrix(const glm::vec3 vec)
{
    float* data = (float*)glm::value_ptr(vec);
    printf("[%f %f %f]\n", data[0], data[1], data[2]);
}

__device__ inline void PrintMatrix(const glm::mat3 mat)
{   // Column Major
    float* data = (float*)glm::value_ptr(mat);
    printf("[");
    for(int i = 0; i < 3; i++){
        printf("[");
        for(int j = 0; j < 3; j++){
            printf("%f ", data[j * 3 + i]);
        }
        printf("]");
        if(i < 2)
            printf("\n");
    }
    printf("]\n");
}


__global__ void EWAProjectForwardCUDAKernel(
    int P,
    const dim3 grid,
    const float* cov3d,
    const float* viewmat,
    const float* camparam,
    const float* uv,
    const float* depth,
    const float* xy,
    const int W, const int H,
    const bool* visibility_status,
    float* cov2d,
    int* radii,
    int* tiles_touched
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visibility_status[idx])
        return;

    float focal_x = camparam[0];
    float focal_y = camparam[1];

    float3 t = {uv[2*idx], uv[2*idx+1], depth[idx]};

    printf("%f %f %f\n", t.x, t.y, t.z);

	const float limx = 1.3f * 0.5 * W / focal_x;
	const float limy = 1.3f * 0.5 * H / focal_y;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

    // EWA project
    // float3 t = {uv[2*idx], uv[2*idx+1], depth[idx]};
    // t.x = t.x * t.z;
    // t.y = t.y * t.z;

    glm::mat3 Jmat = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);
    
    printf("MY J:\n");
    PrintMatrix(Jmat);

    glm::mat3 Wmat = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    printf("MY W:\n");
    PrintMatrix(Wmat);

    glm::mat3 Tmat = Wmat * Jmat;

    printf("MY T:\n");
    PrintMatrix(Tmat);

    glm::mat3 Vrk = glm::mat3(
        cov3d[6 * idx + 0], cov3d[6 * idx + 1], cov3d[6 * idx + 2],
        cov3d[6 * idx + 1], cov3d[6 * idx + 3], cov3d[6 * idx + 4],
        cov3d[6 * idx + 2], cov3d[6 * idx + 4], cov3d[6 * idx + 5]);
    
    printf("MY Vrk:\n");
    PrintMatrix(Vrk);

    glm::mat3 ewa_cov = glm::transpose(Tmat) * glm::transpose(Vrk) * Tmat;

    float3 cov = { float(ewa_cov[0][0]) + 0.3f, float(ewa_cov[0][1]), float(ewa_cov[1][1]) + 0.3f};

    printf("MY cov: %f, %f, %f\n", cov.x, cov.y, cov.z);
    
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
    float2 mean2D = { xy[2 * idx], xy[2 * idx + 1] };
    getRect(mean2D, my_radius, rect_min, rect_max, grid);

    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // cov2d 
    float det_inv = 1.f / det;
    cov2d[3 * idx] = cov.z * det_inv;
    cov2d[3 * idx + 1] = -cov.y * det_inv;
    cov2d[3 * idx + 2] = cov.x * det_inv;

    // radius
    radii[idx] = my_radius;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


__global__ void EWAProjectBackCUDAKernel(
    int P,
    const float* cov3d,
    const float* viewmat,
    const float* camparam,
    const float* uv,
    const float* depth,
    const float* xy,
    const int W, const int H,
    const bool* visibility_status,
    const float* dL_dconics,
    float* dL_dcov3d,
    float* dL_dd
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visibility_status[idx])
        return;

    const float* cov3D = cov3d + 6 * idx;

    float focal_x = camparam[0];
    float focal_y = camparam[1];

    // Fetch gradients, recompute 2D covariance and relevant 
    // intermediate forward results needed in the backward.
    float3 dL_dconic = {dL_dconics[3*idx+0], dL_dconics[3*idx+1], dL_dconics[3*idx+2]};
    float3 t = {
        uv[2*idx+0] * depth[idx],
        uv[2*idx+1] * depth[idx],
        depth[idx]
    };

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 Wmat = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 T = Wmat * J;

    glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Use helper variables for 2D covariance entries. More compact.
    float a = cov2D[0][0] += 0.3f;
    float b = cov2D[0][1];
    float c = cov2D[1][1] += 0.3f;

    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;
    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    printf("%.12f\n", denom2inv);

    if (denom2inv != 0)
    {
        printf("%.12f\n", denom2inv);
        
        float* dL_dcov3D = dL_dcov3d + 6 * idx;
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
        // given gradients w.r.t. 2D covariance matrix (diagonal).
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov3D[0] += (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
        dL_dcov3D[3] += (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
        dL_dcov3D[5] += (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
        // given gradients w.r.t. 2D covariance matrix (off-diagonal).
        // Off-diagonal elements appear twice --> double the gradient.
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov3D[1] += 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
        dL_dcov3D[2] += 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
        dL_dcov3D[4] += 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
    }

    // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
    // cov2D = transpose(T) * transpose(Vrk) * T;
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

    // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
    // T = W * J
    float dL_dJ00 = Wmat[0][0] * dL_dT00 + Wmat[0][1] * dL_dT01 + Wmat[0][2] * dL_dT02;
    float dL_dJ02 = Wmat[2][0] * dL_dT00 + Wmat[2][1] * dL_dT01 + Wmat[2][2] * dL_dT02;
    float dL_dJ11 = Wmat[1][0] * dL_dT10 + Wmat[1][1] * dL_dT11 + Wmat[1][2] * dL_dT12;
    float dL_dJ12 = Wmat[2][0] * dL_dT10 + Wmat[2][1] * dL_dT11 + Wmat[2][2] * dL_dT12;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor &visibility_status
){
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(xy);
    CHECK_INPUT(visibility_status);

    const int P = uv.size(0);
    auto float_opts = uv.options().dtype(torch::kFloat32);
     auto int_opts = uv.options().dtype(torch::kInt32);

    torch::Tensor cov2d = torch::zeros({P, 3}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);

    if(P != 0){
        const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

        EWAProjectForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
            P, 
            tile_grid,
            cov3d.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            xy.contiguous().data_ptr<float>(),
            W, H,
            visibility_status.contiguous().data_ptr<bool>(),
            cov2d.data_ptr<float>(),
            radii.data_ptr<int>(),
            tiles_touched.data_ptr<int>()
        );
    }

    return std::make_tuple(cov2d, radii, tiles_touched);
}

std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor& visibility_status,
    const torch::Tensor& dL_dcov2d
){
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(xy);
    CHECK_INPUT(visibility_status);
    CHECK_INPUT(dL_dcov2d);

    const int P = uv.size(0);
    auto float_opts = uv.options().dtype(torch::kFloat32);
    torch::Tensor dL_dcov3d = torch::zeros({P, 6}, float_opts);
    torch::Tensor dL_dd = torch::zeros({P, 1}, float_opts);

    if(P != 0){
        // Kernel
        EWAProjectBackCUDAKernel<<<(P + 255) / 256, 256 >>>(
            P, 
            cov3d.contiguous().data_ptr<float>(), 
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            xy.contiguous().data_ptr<float>(),
            W, H,
            visibility_status.contiguous().data_ptr<bool>(),
            dL_dcov2d.contiguous().data_ptr<float>(),
            dL_dcov3d.data_ptr<float>(),
            dL_dd.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dcov3d, dL_dd);
}