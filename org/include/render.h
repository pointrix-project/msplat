#ifndef CUDA_RENDER_H_INCLUDED
#define CUDA_RENDER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda_runtime_api.h>


int RenderForwardCUDA(
	const int P,
	const int width, const int height,
	const float* features,
	const float* depths,
	const int32_t* radii,
	const float2* means2D,
	const float3* conics,
	const float* opacities,
	const int32_t* tiles_touched,
	const bool* visibility_filter,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	float* out_feature,
	bool debug);

void RenderBackwardCUDA(
	const int P, const int num_rendered,
	const int width, const int height,
	const float* features,
	const float2* means2D,
	const float3* conics,
	const float* opacities,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dout_feature,
	float* dL_dfeatures,
	float2* dL_dmeans2D,
	float3* dL_dconic,
	float* dL_dopacity,
	bool debug);

#endif