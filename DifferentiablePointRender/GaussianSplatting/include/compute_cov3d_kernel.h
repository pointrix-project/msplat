
#ifndef CUDA_COMPUTE_COV3D_H_INCLUDED
#define CUDA_COMPUTE_COV3D_H_INCLUDED

void computeCov3DForwardCUDA(
	const int P, 
	const float* scales,
	const float* uquats,
	const bool* visibility_status,
	float* cov3Ds);

void computeCov3DBackwardCUDA(
	const int P, 
	const float* scales,
	const float* uquats,
	const bool* visibility_status,
	const float* dL_dcov3Ds,
	float* dL_dscales,
	float* dL_duquats
);

#endif