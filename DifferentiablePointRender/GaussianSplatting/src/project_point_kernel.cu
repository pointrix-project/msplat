/**
 * @file project_point_kernel.cu
 * @author Jian Gao
 * @brief 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <project_point_kernel.h>

namespace cg = cooperative_groups;


/**
 * @brief CUDA kernel for projecting points onto the screen in a forward pass.
 * 
 * @param[in] P             Number of points
 * @param[in] xyz           Coordinates of the points
 * @param[in] viewmat       Camera's view matrix, transforms points from world coordinates to camera coordinates
 * @param[in] projmat       Camera's projection matrix, transforms points from world coordinates to normalized device coordinates (NDC)
 * @param[in] camparam      Camera parameters, [fx, fy, cx, cy]
 * @param[in] W             Width of the image
 * @param[in] H             Height of the image
 * @param[out] uv           Coordinates of the points on the image
 * @param[out] depths       Depths of the points
 * @param[in] nearest       Culling points with depth less than this value
 * @param[in] extent        Culling points outside the extent
 */
__global__ void projectPointForwardCUDAKernel(
    const int P, 
    const float* xyz, 
    const float* viewmat,
    const float* projmat,
    const float* camparam,
    const int W, const int H,
    float2* uv,
    float* depths,
    const float nearest,
    const float extent)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 pt = { xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]};
    
    float4 p_hom = transform_point_4x4(projmat, pt);
	float p_w = 1.0f / (p_hom.w + 1e-7);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    float3 p_view = transform_point_4x3(viewmat, pt);

    bool near_culling = p_view.z <= nearest;
    bool extent_culling = p_proj.x < -extent || p_proj.x > extent || p_proj.y < -extent || p_proj.y > extent;
    if (near_culling || extent_culling)
        return ;

    uv[idx] = { p_proj.x, p_proj.y};
    depths[idx] = p_view.z;
}


__global__ void projectPointBackwardCUDAKernel(
    const int P, 
    const float* xyz, 
    const float* viewmat,
    const float* projmat,
    const float* camparam,
    const int W, const int H,
    const float2* uv,
    const float* depth,
    const float2* dL_duv,
    const float* dL_ddepth,
    glm::vec3* dL_dxyz)
{
    auto idx = cg::this_grid().thread_rank();

    // depth == 0 means culled
    if (idx >= P || depth[idx] == 0)
        return;

    float3 pt = { xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]};

    glm::vec3 dL_dxyz_idx;
    dL_dxyz_idx.x = dL_ddepth[idx] * viewmat[2];
    dL_dxyz_idx.y = dL_ddepth[idx] * viewmat[6];
    dL_dxyz_idx.z = dL_ddepth[idx] * viewmat[10];

    // dpw_dx
    float4 p_hom = transform_point_4x4(projmat, pt);
    float pw = 1.0f / (p_hom.w + 1e-7);

    glm::vec3 dpw_dxyz;
    dpw_dxyz.x = - pw * pw * projmat[3];
    dpw_dxyz.y = - pw * pw * projmat[7];
    dpw_dxyz.z = - pw * pw * projmat[11];

    dL_dxyz_idx.x += dL_duv[idx].x * (projmat[0] * pw + p_hom.x * dpw_dxyz.x);
    dL_dxyz_idx.y += dL_duv[idx].x * (projmat[4] * pw + p_hom.x * dpw_dxyz.y);
    dL_dxyz_idx.z += dL_duv[idx].x * (projmat[8] * pw + p_hom.x * dpw_dxyz.z);

    dL_dxyz_idx.x += dL_duv[idx].y * (projmat[1] * pw + p_hom.y * dpw_dxyz.x);
    dL_dxyz_idx.y += dL_duv[idx].y * (projmat[5] * pw + p_hom.y * dpw_dxyz.y);
    dL_dxyz_idx.z += dL_duv[idx].y * (projmat[9] * pw + p_hom.y * dpw_dxyz.z);

    dL_dxyz[idx] = dL_dxyz_idx;
}

void projectPointForwardCUDA(
    const int P, 
    const float* xyz, 
    const float* viewmatrix,
    const float* projmatrix,
    const float* camparam,
    const int W, const int H,
    float2* uv,
    float* depths,
    const float nearest,
    const float extent)
{
    projectPointForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
        P, 
        xyz, 
        viewmatrix, 
        projmatrix,
        camparam,
        W, H,
        uv,
        depths,
        nearest,
        extent
    );
}


void projectPointBackwardCUDA(
    const int P, 
    const float* xyz, 
    const float* viewmat,
    const float* projmat,
    const float* camparam,
    const int W, const int H,
    const float2* uv,
    const float* depth,
    const float2* dL_duv,
    const float* dL_ddepth,
    float* dL_dxyz)
{
    projectPointBackwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
        P, 
        xyz, 
        viewmat, 
        projmat,
        camparam,
        W, H,
        uv,
        depth,
        dL_duv,
        dL_ddepth,
        (glm::vec3*)dL_dxyz
    );
}
