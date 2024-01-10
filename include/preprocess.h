#ifndef CUDA_PREPROCESS_H_INCLUDED
#define CUDA_PREPROCESS_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda_runtime_api.h>


void PreprocessForwardCUDA(
	const int P, 
	const int width, const int height,
	const float* means3D,
	const float* scales,
	const float* rotations,
	const float* cov3Ds_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, const float tan_fovy,
	float* depths,
	int32_t* radii,
	float2* means2D,
	float* cov3Ds,
	float3* conics,
	int32_t* tiles_touched,
	const bool debug);

void PreprocessBackwardCUDA(
	const int P, 
	const int width, int height,
	const float3* means3D,
	const float* scales,
	const float* rotations,
	const float* cov3Ds,
	const float* cov3Ds_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	const float* dL_ddepths,
	const float2* dL_dmeans2D,
	const float3* dL_dconics,
	float* dL_dmeans3D,
	float* dL_dscales,
	float4* dL_drotations,
	float* dL_dcov3Ds,  // need to add grad on this tensor, because cov3Ds is also a output in forward (while not consider the effect from cov2D)
	bool debug);

#endif