#ifndef CUDA_COV_H_INCLUDED
#define CUDA_COV_H_INCLUDED

#include <glm/glm.hpp>

void computeCov3DForwardCUDA(
	const int P, 
	const float* scales,
	const float* rotations,
	const bool* visibility_filter,
	float* cov3Ds);

void computeCov3DBackwardCUDA(
	const int P, 
	const float* scales,
	const float* rotations,
	const bool* visibility_filter,
	const float* dL_dcov3Ds,
	float* dL_dscales,
	float* dL_drotations
);

#endif