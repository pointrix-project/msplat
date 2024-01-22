
#ifndef CUDA_COMPUTE_SH_KERNEL_H_INCLUDED
#define CUDA_COMPUTE_SH_KERNEL_H_INCLUDED

void computeSHForwardCUDA(
    const int P,
    const float* shs,
    const int deg,
    const float* viewdirs,
    const bool* visibility_status,
    float* colors,
    bool* clamped
);

void computeSHBackwardCUDA(
    const int P, 
    const float* shs,
    const int deg,
    const float* dirs,
    const bool* visibility_status,
    const bool* clamped,
    const float* dL_dcolors,
    float* dL_dshs,
    float* dL_ddirs
);

#endif