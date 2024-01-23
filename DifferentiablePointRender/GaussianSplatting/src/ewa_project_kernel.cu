
#include <config.h>
#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <ewa_project_kernel.h>


namespace cg = cooperative_groups;


// Params change viewmat, projmat
__global__ void EWAProjectForwardCUDAKernel(
    int P,
    const dim3 grid,
    const float* xyz,
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

    // EWA project
    float3 t = {uv[2*idx], uv[2*idx+1], depth[idx]};
    t.x = t.x * t.z;
    t.y = t.y * t.z;

    glm::mat3 Jmat = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 Wmat = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    glm::mat3 Tmat = Wmat * Jmat;
    glm::mat3 Vrk = glm::mat3(
        cov3d[6 * idx + 0], cov3d[6 * idx + 1], cov3d[6 * idx + 2],
        cov3d[6 * idx + 1], cov3d[6 * idx + 3], cov3d[6 * idx + 4],
        cov3d[6 * idx + 2], cov3d[6 * idx + 4], cov3d[6 * idx + 5]);

    glm::mat3 ewa_cov = glm::transpose(Tmat) * glm::transpose(Vrk) * Tmat;

    float3 cov = { float(ewa_cov[0][0]) + 0.3f, float(ewa_cov[0][1]), float(ewa_cov[1][1]) + 0.3f};
    
    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles. 
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    uint2 rect_min, rect_max;
    float2 mean2D = { xy[2*idx], xy[2*idx+1] };
    getRect(mean2D, my_radius, rect_min, rect_max, grid);

    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    cov2d[3*idx] = cov.z * det_inv;
    cov2d[3*idx+1] = -cov.y * det_inv;
    cov2d[3*idx+2] = cov.x * det_inv;

    // radius
    radii[idx] = my_radius;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// __global__ void EWAProjectBackwardCUDAKernel(
//     int P,
//     const float* xyz,
//     const float* cov3d,
//     const float* viewmat,
//     const float* projmat,
//     const float* camparam,
//     const int W, const int H,
//     const bool* visibility_status,
//     const float* cov2d,
//     const float* opacity,
//     float* dL_dxyz,
//     float* dL_dcov3d
// ){

// }


void EWAProjectForwardCUDA(
    int P,
    const float* xyz,
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

    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    EWAProjectForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
        P, 
        tile_grid,
        xyz, 
        cov3d,
        viewmat,
        camparam,
        uv,
        depth,
        xy,
        W, H,
        visibility_status,
        cov2d,
        radii,
        tiles_touched
    );
}

// void EWAProjectBackwardCUDA(
//     int P,
//     const float* xyz,
//     const float* cov3d,
//     const float* viewmat,
//     const float* projmat,
//     const float* camparam,
//     const int W, const int H,
//     const bool* visibility_status,
//     const float* cov2d,
//     const float* opacity,
//     float* dL_dxyz,
//     float* dL_dcov3d
// ){
//     // EWAProjectBackCUDAKernel <<<(P + 255) / 256, 256 >>> (
//     //     P, 
//     //     xyz,
//     //     cov3d,
//     //     viewmat, 
//     //     projmat,
//     //     camparam,
//     //     W, H,
//     //     visibility_status
//     //     cov2d,
//     //     opacity,
//     //     dL_dxyz,
//     //     dL_dcov3d
//     // );
// }