/**
 * @file project_point_kernel.h
 * @author Jian Gao
 * @brief 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CUDA_PROJECT_POINT_KERNEL_H_INCLUDED
#define CUDA_PROJECT_POINT_KERNEL_H_INCLUDED

#include <cuda_runtime_api.h>

/**
 * @brief Wrapper function for launching the CUDA kernel to project points onto the screen in a forward pass. 
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
 * @param[in] extent         Culling points outside the extent
 */
void projectPointForwardCUDA(
    const int P, 
    const float* xyz, 
    const float* viewmat,
    const float* projmat,
    const float* camparam,
    const int W, const int H,
    float2* uv,
    float* depths,
    const float nearest,
    const float extent);


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
    float* dL_dxyz);

# endif