#ifndef CUDA_SH_H_INCLUDED
#define CUDA_SH_H_INCLUDED

void ComputeColorFromSHForwardCUDA(
    const int P, const int deg, const int max_coeffs,
    const float* dirs,
    const float* shs,
    const bool* visibility_filter,
    float* colors,
    bool* clamped
);

void ComputeColorFromSHBackwardCUDA(
    const int P, const int deg, const int max_coeffs,
    const float* dirs,
    const float* shs,
    const bool* visibility_filter,
    const bool* clamped,
	const float* dL_dcolors,
	float* dL_ddirs,
	float* dL_dshs
);

#endif