/**
 * @file utils.h
 * @brief
 */

#pragma once

// Check the input, magic lines.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x)

#include <config.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__forceinline__ __device__ void get_rect(const float2 p,
                                         int max_radius,
                                         uint2 &rect_min,
                                         uint2 &rect_max,
                                         dim3 grid) {

    rect_min.x = thrust::min<int>(
        grid.x,
        thrust::max<int>(0, static_cast<int>((p.x - max_radius) / BLOCK_X)));
    rect_min.y = thrust::min<int>(
        grid.y,
        thrust::max<int>(0, static_cast<int>((p.y - max_radius) / BLOCK_Y)));
    rect_max.x = thrust::min<int>(
        grid.x,
        thrust::max<int>(
            0, static_cast<int>((p.x + max_radius + BLOCK_X - 1) / BLOCK_X)));
    rect_max.y = thrust::min<int>(
        grid.y,
        thrust::max<int>(
            0, static_cast<int>((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)));
}