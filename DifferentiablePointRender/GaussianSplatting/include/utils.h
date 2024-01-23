
#pragma once

// Check the input, magic lines.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x)


#include <config.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>


__forceinline__ __device__ float ndc_to_pixel(float ndc, int size)
{
    return ((ndc + 1.0) * size - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transform_point_4x3(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}

__forceinline__ __device__ float4 transform_point_4x4(const float* matrix, const float3& p)
{
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };

    return transformed;
}


__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
    rect_min.x = thrust::min<int>(grid.x, thrust::max<int>(0, static_cast<int>((p.x - max_radius) / BLOCK_X)));
    rect_min.y = thrust::min<int>(grid.y, thrust::max<int>(0, static_cast<int>((p.y - max_radius) / BLOCK_Y)));
    rect_max.x = thrust::min<int>(grid.x, thrust::max<int>(0, static_cast<int>((p.x + max_radius + BLOCK_X - 1) / BLOCK_X)));
    rect_max.y = thrust::min<int>(grid.y, thrust::max<int>(0, static_cast<int>((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)));
}
